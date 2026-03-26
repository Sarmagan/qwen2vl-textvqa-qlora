"""
evaluate.py — Baseline vs Fine-Tuned Qwen2-VL-2B on TextVQA

Usage:
    python evaluate.py --adapter_path ./qwen2vl_textvqa_qlora --num_samples 500 --batch_size 4
"""

import argparse, json, os, re, string, sys
from pathlib import Path
from typing import Optional

# ── Imports ───────────────────────────────────────────────────────────────────
import torch
from tqdm import tqdm
from PIL import Image

from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
from peft import PeftModel
from datasets import load_dataset
from qwen_vl_utils import process_vision_info

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--adapter_path", default="./qwen2vl_textvqa_qlora",
                    help="Path to the saved LoRA adapter directory")
parser.add_argument("--num_samples", type=int, default=500,
                    help="Number of test samples to evaluate")
parser.add_argument("--batch_size", type=int, default=4,
                    help="Inference batch size (per model)")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--results_path", default="./eval_results.json")
parser.add_argument(
    "--eval_holdout_start",
    type=int,
    default=None,
    help="Index into shuffled validation where holdout eval starts (default: val_samples from run_meta.json, else 500). "
    "Must match training so eval does not reuse the Trainer validation set.",
)
parser.add_argument(
    "--no_short_answer_suffix",
    action="store_true",
    help="Omit the short-answer instruction suffix (use if your adapter was trained before that suffix existed in train.py).",
)
args = parser.parse_args()

# Must match train.py so inference distribution aligns with fine-tuning.
_DEFAULT_ANSWER_SUFFIX = "\nAnswer with a short phrase only (a few words)."


# ── Helpers ───────────────────────────────────────────────────────────────────
def normalize(s: str) -> str:
    """Lower-case, strip punctuation and extra whitespace."""
    s = s.lower().strip()
    s = s.translate(str.maketrans("", "", string.punctuation))
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> bool:
    return pred.strip() == gold.strip()


def relaxed_match(pred: str, gold: str) -> bool:
    return normalize(pred) == normalize(gold)


def vqa_accuracy(pred: str, answers: list) -> float:
    """
    Official VQA accuracy: min(#human answers matching pred / 3, 1.0)
    TextVQA provides up to 10 answers per question.
    """
    norm_pred = normalize(pred)
    hits = sum(normalize(a) == norm_pred for a in answers)
    return min(hits / 3.0, 1.0)


def substring_hit(pred: str, answers: list) -> bool:
    """True if any normalized reference answer appears as a substring of pred (for verbose generations)."""
    np = normalize(pred)
    return any(normalize(a) in np and len(normalize(a)) > 0 for a in answers)


# ── Model loader ──────────────────────────────────────────────────────────────
def load_model(base_id: str, adapter_path: Optional[str] = None):
    """
    Loads the base model in 4-bit. If adapter_path is given, applies LoRA.
    Returns (model, processor).
    """
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        base_id,
        quantization_config=bnb_cfg,
        device_map="auto",
        dtype=torch.float16,
        trust_remote_code=True,
    )

    if adapter_path is not None:
        print(f"  → Applying LoRA adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()   # merge for faster inference

    model.eval()

    proc_path = adapter_path if adapter_path else base_id
    processor = AutoProcessor.from_pretrained(
        proc_path,
        min_pixels=256 * 28 * 28,
        max_pixels=512 * 28 * 28,
        use_fast=False,
    )
    return model, processor


# ── Batch inference ───────────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(
    model, processor, samples, batch_size: int, answer_suffix: str = _DEFAULT_ANSWER_SUFFIX
) -> list[str]:
    """Returns a list of predicted answer strings."""
    predictions = []
    device = next(model.parameters()).device

    for start in tqdm(range(0, len(samples), batch_size),
                      desc="  Inference", leave=False):
        batch = samples[start: start + batch_size]
        texts, flat_images = [], []

        for item in batch:
            image    = item["image"]
            question = item["question"]

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text",  "text": question + answer_suffix},
                    ],
                }
            ]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            imgs, _ = process_vision_info(messages)
            texts.append(text)
            flat_images.extend(imgs or [])

        inputs = processor(
            text=texts,
            images=flat_images if flat_images else None,
            padding=True,
            truncation=False,
            return_tensors="pt",
        ).to(device)

        tok = processor.tokenizer
        out_ids = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )

        # Strip the prompt tokens; only decode the generated part
        gen_ids = [
            out[len(inp):]
            for out, inp in zip(out_ids, inputs["input_ids"])
        ]
        decoded = processor.batch_decode(
            gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        # First line only — models sometimes continue with extra sentences.
        predictions.extend([d.strip().split("\n")[0].strip() for d in decoded])

    return predictions


# ── Evaluation loop ───────────────────────────────────────────────────────────
def evaluate_model(model, processor, samples, batch_size: int, label: str):
    print(f"\n{'─'*50}")
    print(f"  Evaluating: {label}")
    print(f"{'─'*50}")

    preds = run_inference(
        model,
        processor,
        samples,
        batch_size,
        answer_suffix="" if args.no_short_answer_suffix else _DEFAULT_ANSWER_SUFFIX,
    )

    em_scores, rm_scores, vqa_scores, sub_scores = [], [], [], []
    for pred, item in zip(preds, samples):
        answers = item["answers"] if isinstance(item["answers"], list) else [item["answers"]]
        gold    = answers[0]

        em_scores.append(float(exact_match(pred, gold)))
        rm_scores.append(float(relaxed_match(pred, gold)))
        vqa_scores.append(vqa_accuracy(pred, answers))
        sub_scores.append(float(substring_hit(pred, answers)))

    n = len(samples)
    results = {
        "label":              label,
        "n_samples":          n,
        "exact_match":        round(sum(em_scores)  / n * 100, 2),
        "relaxed_match":      round(sum(rm_scores)  / n * 100, 2),
        "vqa_accuracy":       round(sum(vqa_scores) / n * 100, 2),
        "substring_hit":      round(sum(sub_scores) / n * 100, 2),
        "sample_predictions": [
            {
                "question": samples[i]["question"],
                "gold":     samples[i]["answers"][0] if isinstance(samples[i]["answers"], list)
                            else samples[i]["answers"],
                "pred":     preds[i],
            }
            for i in range(min(10, n))   # save first 10 for inspection
        ],
    }

    print(f"  Exact Match    : {results['exact_match']:6.2f}%")
    print(f"  Relaxed Match  : {results['relaxed_match']:6.2f}%")
    print(f"  VQA Accuracy   : {results['vqa_accuracy']:6.2f}%")
    print(f"  Substring hit  : {results['substring_hit']:6.2f}%  (any ref answer ⊂ pred; for long outputs)")
    return results, preds


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*60}")
    print("  Qwen2-VL-2B  —  Baseline vs Fine-Tuned Evaluation")
    print(f"{'='*60}\n")

    # Read base model id and eval slice from adapter meta if available
    base_id = "Qwen/Qwen2-VL-2B-Instruct"
    meta: dict = {}
    meta_path = os.path.join(args.adapter_path, "run_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        base_id = meta.get("base_model", base_id)

    # ── Load holdout eval data (disjoint from training-time validation) ───────
    print("[1/4] Loading TextVQA validation (holdout slice) …")
    raw = load_dataset("textvqa", trust_remote_code=True)
    val_full = raw["validation"]
    n_val = len(val_full)
    # Same ordering as train.py: one shuffle; Trainer used rows [0, val_samples).
    shuffle_seed = meta.get("validation_shuffle_seed", meta.get("seed", args.seed))
    val_used_in_training = meta.get("val_samples", 500)
    holdout_start = (
        args.eval_holdout_start
        if args.eval_holdout_start is not None
        else val_used_in_training
    )
    val_shuffled = val_full.shuffle(seed=shuffle_seed)
    n_take = min(args.num_samples, max(0, n_val - holdout_start))
    if n_take == 0:
        print(
            f"  ERROR: No holdout rows (validation len={n_val}, holdout_start={holdout_start}). "
            "Lower val_samples in training or --eval_holdout_start.",
            file=sys.stderr,
        )
        sys.exit(1)
    test_raw = val_shuffled.select(range(holdout_start, holdout_start + n_take))
    samples = list(test_raw)
    print(
        f"  {len(samples)} samples  |  shuffle_seed={shuffle_seed}  |  "
        f"indices [{holdout_start}, {holdout_start + len(samples)})  "
        f"(Trainer val used [0, {val_used_in_training}) during training)\n"
    )

    all_results = {
        "eval_split": "validation_holdout",
        "validation_shuffle_seed": shuffle_seed,
        "holdout_index_range": [holdout_start, holdout_start + len(samples)],
        "trainer_val_used_range": [0, val_used_in_training],
    }

    # ── Baseline ──────────────────────────────────────────────────────────────
    print("[2/4] Loading baseline model …")
    baseline_model, baseline_proc = load_model(base_id, adapter_path=None)
    baseline_res, baseline_preds = evaluate_model(
        baseline_model, baseline_proc, samples, args.batch_size, "Baseline (no fine-tune)"
    )
    all_results["baseline"] = baseline_res

    # Free VRAM before loading fine-tuned model
    del baseline_model
    torch.cuda.empty_cache()

    # ── Fine-tuned ────────────────────────────────────────────────────────────
    print(f"\n[3/4] Loading fine-tuned model from {args.adapter_path} …")
    ft_model, ft_proc = load_model(base_id, adapter_path=args.adapter_path)
    ft_res, ft_preds = evaluate_model(
        ft_model, ft_proc, samples, args.batch_size, "Fine-Tuned (QLoRA)"
    )
    all_results["fine_tuned"] = ft_res

    # ── Delta summary ─────────────────────────────────────────────────────────
    delta_em  = ft_res["exact_match"]  - baseline_res["exact_match"]
    delta_rm  = ft_res["relaxed_match"]- baseline_res["relaxed_match"]
    delta_vqa = ft_res["vqa_accuracy"] - baseline_res["vqa_accuracy"]
    delta_sub = ft_res["substring_hit"] - baseline_res["substring_hit"]

    print(f"\n{'='*60}")
    print("  RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Metric':<22} {'Baseline':>10} {'Fine-Tuned':>12} {'Delta':>8}")
    print(f"  {'─'*54}")
    print(f"  {'Exact Match':<22} {baseline_res['exact_match']:>9.2f}%"
          f" {ft_res['exact_match']:>11.2f}%  {delta_em:>+7.2f}pp")
    print(f"  {'Relaxed Match':<22} {baseline_res['relaxed_match']:>9.2f}%"
          f" {ft_res['relaxed_match']:>11.2f}%  {delta_rm:>+7.2f}pp")
    print(f"  {'VQA Accuracy':<22} {baseline_res['vqa_accuracy']:>9.2f}%"
          f" {ft_res['vqa_accuracy']:>11.2f}%  {delta_vqa:>+7.2f}pp")
    print(f"  {'Substring hit':<22} {baseline_res['substring_hit']:>9.2f}%"
          f" {ft_res['substring_hit']:>11.2f}%  {delta_sub:>+7.2f}pp")
    print(f"{'='*60}\n")

    all_results["delta"] = {
        "exact_match":   round(delta_em,  2),
        "relaxed_match": round(delta_rm,  2),
        "vqa_accuracy":  round(delta_vqa, 2),
        "substring_hit": round(delta_sub, 2),
    }

    # ── Per-sample comparison (first 10) ──────────────────────────────────────
    print("  Sample predictions (first 10):")
    print(f"  {'Q':<40} {'Gold':<15} {'Base':<15} {'FT':<15}")
    print(f"  {'─'*85}")
    for i in range(min(10, len(samples))):
        q    = samples[i]["question"][:38]
        gold = (samples[i]["answers"][0] if isinstance(samples[i]["answers"], list)
                else samples[i]["answers"])[:13]
        base = baseline_preds[i][:13]
        ft   = ft_preds[i][:13]
        print(f"  {q:<40} {gold:<15} {base:<15} {ft:<15}")

    # ── Save results ──────────────────────────────────────────────────────────
    print(f"\n[4/4] Saving results to {args.results_path} …")
    with open(args.results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print("Done!\n")


if __name__ == "__main__":
    main()