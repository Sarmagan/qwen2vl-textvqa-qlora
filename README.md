# Qwen2-VL on TextVQA (QLoRA)

This repo adapts **Qwen2-VL-2B-Instruct** to **TextVQA**—reading text in images and answering short questions about it. We fine-tune with **QLoRA** (4-bit base model + small LoRA adapters on the language model; vision stays frozen), then compare the **base checkpoint** and the **adapter** on the same held-out questions so gains are apples-to-apples.

### Example data point (TextVQA)

One row from the **train** split ([`textvqa` on Hugging Face](https://huggingface.co/datasets/textvqa)): a photo plus a question whose answer is grounded in on-image text.

![TextVQA example: phone with visible branding](assets/textvqa_example.png)

| | |
|--|--|
| **Question** | what is the brand of phone? |
| **Reference answers** (annotators; subset) | nokia, nokia, nokia, … |

---

## Results

Evaluation uses the full **non-overlapping** TextVQA validation holdout (no overlap with the subset used for Trainer validation during training). `evaluate.py` scores each model on the same questions; the README highlights the main metric, **VQA accuracy**.

| Metric | Baseline | Fine-tuned | Δ |
|--------|----------|------------|---|
| VQA accuracy | 75.19% | 78.10% | +2.91 pp |

### Metrics (how they are computed)

- **VQA accuracy** — Standard VQA scoring: TextVQA provides up to **10** reference answers per question. After the same normalization, count how many references equal the prediction; the score for that question is **min(hits / 3, 1)**. This rewards agreement with multiple annotators and is **not** tied to only the first reference. Reported as the **average** of these per-question scores, shown as a percentage.

`evaluate.py` still reports additional diagnostics such as exact match and relaxed match for deeper analysis.

---

## Setup

- **GPU** with enough VRAM for 4-bit Qwen2-VL at your batch settings (defaults target roughly **~11 GB**; reduce image pixel bounds or batch size if needed).
- **Python** 3.9+ (newer Python makes it easier to install recent `bitsandbytes` wheels).

```bash
pip install "torch" --index-url https://download.pytorch.org/whl/cu118   # match your CUDA
pip install "transformers>=4.45.0" accelerate peft datasets tqdm pillow bitsandbytes qwen-vl-utils
```

Hugging Face Hub access for **TextVQA** and **Qwen2-VL** weights (`huggingface-cli login` if a model is gated).

---

## Train

```bash
python train.py
# Multi-GPU:
torchrun --nproc_per_node=2 train.py
```

Training pulls TextVQA from the Hub, formats chats with image + question, and optimizes only the assistant answer tokens. By default it uses the full TextVQA train split. Each step uses **one row per question**; the gold string is sampled from the unique reference pool on the train split (`answer_sampling`: `random` per step, or `cycle` across epochs). Validation uses a **deterministic** target (`val_answer_sampling`: `first` reference in the pool, or `cycle`) so `eval_loss` and `load_best_model_at_end` are not noisy. Checkpoints, the LoRA adapter, processor, and `run_meta.json` land under `./qwen2vl_textvqa_qlora` by default. Key settings (model id, **`train_samples` / `val_samples`**, `max_answers_per_question`, `answer_sampling`, `val_answer_sampling`, LoRA rank, epochs, LR, batching) live in the `Cfg` dataclass in `train.py`.

---

## Evaluate

```bash
python evaluate.py --adapter_path ./qwen2vl_textvqa_qlora --batch_size 4 --results_path ./eval_results.json
```

By default, evaluation uses the full remaining validation holdout after the training-time validation slice. `run_meta.json` keeps validation shuffle and slice sizes aligned with training so the holdout set stays disjoint from Trainer eval.

---

## Outputs

| Path | Role |
|------|------|
| `output_dir/` | Adapter, processor, Trainer checkpoints |
| `output_dir/run_meta.json` | Seeds and sample counts for reproducible eval |
| `eval_results.json` | Baseline vs fine-tuned metrics and sample predictions |

---

## Troubleshooting

- **OOM** — Lower batch size, raise gradient accumulation, or reduce `max_pixels` in `train.py` where the processor is built.
- **Multi-GPU** — Prefer `torchrun` or a single visible GPU; see comments in `train.py` if you hit 4-bit + DataParallel issues on older stacks.
