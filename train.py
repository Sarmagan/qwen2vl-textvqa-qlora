"""
Qwen2-VL-2B QLoRA Fine-Tuning on TextVQA (subset)
=====================================================
Run (single GPU):   python train.py
Run (2x GPU DDP):   torchrun --nproc_per_node=2 train.py
"""

# ── 1. Imports ────────────────────────────────────────────────────────────────
import os, json, random
import importlib.metadata
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from torch.utils.data import Dataset
from PIL import Image

from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import load_dataset
from qwen_vl_utils import process_vision_info


def _patch_accelerate_dispatch_for_bnb_4bit():
    """
    bitsandbytes<0.43.2 does not support model.to() on 4-bit weights. Accelerate only sets
    force_hooks when model.is_loaded_in_4bit is True, but Transformers sets that flag in
    quantizer.postprocess *after* dispatch_model — so a single-GPU map hits model.to()
    and crashes. PyPI only ships bitsandbytes up to 0.42 for Python 3.9.

    We must patch the *same* function object Transformers calls: it does
    `from accelerate import dispatch_model` at import time, so replacing only
    accelerate.big_modeling.dispatch_model leaves modeling_utils.dispatch_model stale.
    """
    try:
        from packaging.version import Version

        if Version(importlib.metadata.version("bitsandbytes")) >= Version("0.43.2"):
            return
    except (importlib.metadata.PackageNotFoundError, ValueError):
        pass

    import accelerate
    import accelerate.big_modeling as bm
    import transformers.modeling_utils as mu

    _orig = bm.dispatch_model

    def _dispatch(model, **kwargs):
        dm = kwargs.get("device_map")
        # Single-device maps take the branch that calls model.to(); 4-bit + old bnb cannot.
        if dm is not None and len(set(dm.values())) == 1 and not kwargs.get("force_hooks"):
            kwargs["force_hooks"] = True
        return _orig(model, **kwargs)

    bm.dispatch_model = _dispatch
    if hasattr(accelerate, "dispatch_model"):
        accelerate.dispatch_model = _dispatch
    if hasattr(mu, "dispatch_model"):
        mu.dispatch_model = _dispatch


def _model_is_bnb_4bit(model) -> bool:
    """True if the underlying PreTrainedModel was loaded in 4-bit (works through PEFT wrappers)."""
    if getattr(model, "is_loaded_in_4bit", False):
        return True
    base = getattr(model, "base_model", None)
    if base is not None:
        inner = getattr(base, "model", base)
        if getattr(inner, "is_loaded_in_4bit", False):
            return True
    return False


def _patch_trainer_no_dataparallel_for_4bit_vlm():
    """
    Trainer._wrap_model uses nn.DataParallel when n_gpu > 1, but only skips this for 8-bit
    models. Qwen2-VL packs all image patches into dim-0; DataParallel splits that tensor
    incorrectly and vision RoPE lengths mismatch (e.g. 3976 vs 4016). Unwrap DP for 4-bit.
    """
    import torch.nn as nn

    _orig = Trainer._wrap_model

    def _wrap_model(self, model, training=True, dataloader=None):
        out = _orig(self, model, training=training, dataloader=dataloader)
        if isinstance(out, nn.DataParallel) and _model_is_bnb_4bit(out.module):
            return out.module
        return out

    Trainer._wrap_model = _wrap_model


# ── 2. Config ─────────────────────────────────────────────────────────────────
@dataclass
class Cfg:
    model_id: str        = "Qwen/Qwen2-VL-2B-Instruct"
    output_dir: str      = "./qwen2vl_textvqa_qlora"
    dataset_name: str    = "textvqa"          # HF dataset id
    train_samples: int   = 20000              # subset of TextVQA train (~34k)
    val_samples: int     = 500
    # LoRA
    lora_r: int          = 16
    lora_alpha: int      = 32
    lora_dropout: float  = 0.05
    # Training
    epochs: int          = 3
    per_device_bs: int   = 2                  # per GPU
    grad_accum: int      = 8                  # effective bs = 2*2*8 = 32
    learning_rate: float = 2e-4
    warmup_ratio: float  = 0.03
    save_steps: int      = 100
    eval_steps: int      = 100
    logging_steps: int   = 20
    fp16: bool           = True               # cu118 supports fp16 well
    seed: int            = 42

cfg = Cfg()

# Nudge the model toward TextVQA-style short answers (not long captions).
TEXTVQA_USER_SUFFIX = "\nAnswer with a short phrase only (a few words)."

# ── 3. Dataset ────────────────────────────────────────────────────────────────
class TextVQADataset(Dataset):
    """
    Wraps the HF TextVQA split into Qwen2-VL chat format.
    Each item: one image + one question → one answer (first answer string).
    """
    def __init__(self, hf_split, processor):
        self.data      = hf_split
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item  = self.data[idx]
        image = item["image"]                    # PIL Image
        question = item["question"]
        # TextVQA provides a list of answers; use the first one
        answer = item["answers"][0] if isinstance(item["answers"], list) else item["answers"]

        # Build Qwen2-VL chat message
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text",  "text": question + TEXTVQA_USER_SUFFIX},
                ],
            },
            {
                "role": "assistant",
                "content": answer,
            },
        ]
        return messages


def collate_fn(batch, processor):
    """
    Custom collator: processes a batch of chat messages into model inputs
    and masks everything before the assistant answer in labels.

    Do not use tokenizer truncation with vision inputs: cutting at max_length
    breaks Qwen2-VL image placeholder alignment. Pad to the longest sequence
    in the batch instead (truncation=False).
    """
    texts        = []
    image_inputs = []

    for messages in batch:
        # Separate user turn (for image processing) from full conversation
        user_messages = [m for m in messages if m["role"] == "user"]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        imgs, _ = process_vision_info(user_messages)
        texts.append(text)
        image_inputs.append(imgs)

    # Flatten image list per sample for the processor
    # processor expects a flat list when images_per_sample can vary
    flat_images = [img for imgs in image_inputs for img in (imgs or [])]

    inputs = processor(
        text=texts,
        images=flat_images if flat_images else None,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )

    labels = inputs["input_ids"].clone()

    # Mask padding tokens
    labels[labels == processor.tokenizer.pad_token_id] = -100

    # Mask image tokens (Qwen2-VL special token ids)
    IMAGE_TOKEN_IDS = [151652, 151653, 151655]
    for tid in IMAGE_TOKEN_IDS:
        labels[labels == tid] = -100

    # Mask the user / system portion — only train on assistant answer
    # We find the assistant token "<|im_start|>assistant\n" and mask before it
    im_start_id = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
    assistant_id = processor.tokenizer.encode("assistant", add_special_tokens=False)[0]

    for i in range(labels.shape[0]):
        ids = inputs["input_ids"][i].tolist()
        # Find last occurrence of im_start + assistant
        last_asst_pos = -1
        for j in range(len(ids) - 1):
            if ids[j] == im_start_id and ids[j + 1] == assistant_id:
                last_asst_pos = j
        if last_asst_pos != -1:
            # +2 to also skip "assistant\n" header token
            labels[i, : last_asst_pos + 2] = -100

    inputs["labels"] = labels
    return inputs


# ── 4. Main ───────────────────────────────────────────────────────────────────
def main():
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    _patch_trainer_no_dataparallel_for_4bit_vlm()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main    = local_rank == 0

    # ── 4a. Load processor ────────────────────────────────────────────────────
    if is_main:
        print(f"\n{'='*60}")
        print(f"  Qwen2-VL-2B  QLoRA Fine-Tuning on TextVQA")
        print(f"{'='*60}\n")
        print("[1/5] Loading processor …")

    processor = AutoProcessor.from_pretrained(
        cfg.model_id,
        min_pixels=256 * 28 * 28,
        max_pixels=512 * 28 * 28,   # conservative for 11 GB VRAM
        use_fast=False,             # fast path needs torch.compiler.is_compiling() (newer PyTorch)
    )

    # ── 4b. QLoRA model ───────────────────────────────────────────────────────
    if is_main:
        print("[2/5] Loading model in 4-bit QLoRA …")

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,   # fp16 for cu118
    )

    _patch_accelerate_dispatch_for_bnb_4bit()

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        cfg.model_id,
        quantization_config=bnb_cfg,
        device_map={"": local_rank},            # one GPU per process for DDP
        dtype=torch.float16,
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # Target the LLM attention + FFN projections (freeze vision tower)
    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_cfg)

    if is_main:
        model.print_trainable_parameters()

    # ── 4c. Dataset ───────────────────────────────────────────────────────────
    if is_main:
        print("[3/5] Loading TextVQA dataset …")

    raw = load_dataset("textvqa", trust_remote_code=True)

    # Deterministic shuffle + subset
    train_raw = raw["train"].shuffle(seed=cfg.seed).select(range(cfg.train_samples))
    val_raw   = raw["validation"].shuffle(seed=cfg.seed).select(range(cfg.val_samples))

    train_ds = TextVQADataset(train_raw, processor)
    val_ds   = TextVQADataset(val_raw,   processor)

    if is_main:
        print(f"   Train: {len(train_ds):,} samples  |  Val: {len(val_ds):,} samples")

    # ── 4d. Training args ─────────────────────────────────────────────────────
    if is_main:
        print("[4/5] Setting up Trainer …")

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.per_device_bs,
        per_device_eval_batch_size=cfg.per_device_bs,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type="cosine",
        fp16=cfg.fp16,
        bf16=False,                             # bf16 not reliable on cu118
        logging_steps=cfg.logging_steps,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to="none",                       # set "tensorboard" if desired
        ddp_find_unused_parameters=False,
        seed=cfg.seed,
        label_names=["labels"],
    )

    # ── 4e. Trainer ───────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=lambda b: collate_fn(b, processor),
    )

    # ── 4f. Train ─────────────────────────────────────────────────────────────
    if is_main:
        print("[5/5] Starting training …\n")

    trainer.train()

    # ── 4g. Save ──────────────────────────────────────────────────────────────
    if is_main:
        print("\nSaving LoRA adapter …")
        trainer.save_model(cfg.output_dir)
        processor.save_pretrained(cfg.output_dir)

        # Also save the config so evaluate.py can find it
        meta = {
            "base_model": cfg.model_id,
            "adapter_path": cfg.output_dir,
            "dataset": cfg.dataset_name,
            "train_samples": cfg.train_samples,
            "val_samples": cfg.val_samples,
            # Same shuffle as evaluate.py: rows [0, val_samples) = Trainer eval;
            # rows [val_samples, …) are reserved for holdout evaluation (no overlap).
            "seed": cfg.seed,
            "validation_shuffle_seed": cfg.seed,
        }
        with open(os.path.join(cfg.output_dir, "run_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        print(f"\nDone! Adapter saved to: {cfg.output_dir}")


if __name__ == "__main__":
    main()