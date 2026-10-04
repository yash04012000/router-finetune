# PRD 5 — Small open model, LoRA / QLoRA fine-tune

Status: Not started · Depends on: PRDs 1-3 · Blocks: PRDs 6-7

## Summary

The core of the project: fine-tune a small open model with LoRA on the train split, produce
calibrated per-intent probabilities, and ship the adapter with a one-command inference example.
Calibrated confidence matters as much as accuracy here, because PRD 6's hybrid is only as good
as the model's ability to know when it's unsure.

Hardware constraint: one RTX 4060 Ti (8 GB) on Windows 11. Everything in this PRD is sized to
fit that card.

## Goals

- `scripts/train_lora.py`: one command, all hyperparameters in `config/train_lora.yaml`, fixed
  seed, logs hardware/library versions/wall time to `results/training/lora_<model>.json`.
- Primary model: **Qwen2.5-1.5B** (Apache-2.0) with a sequence-classification head + LoRA, bf16.
- QLoRA (4-bit NF4) variant as an ablation — either the same model (to show the memory/accuracy
  trade-off) or a 3B model that only fits in 8 GB quantized.
- Temperature-scaled confidences (PRD 3's `calibrate.py`), fit on val.
- Adapter + head pushed to the Hugging Face Hub; `scripts/route.py "message"` downloads and
  predicts in one command.
- Predictions on val and test; batch-1 latency and batched throughput measured.

## Non-goals

- Serving infrastructure (vLLM, batching servers). Measured with plain transformers; DESIGN.md
  discusses what changes in production.
- Large hyperparameter sweeps. A small grid on val, documented.

## Design

### Classification head vs generative labels

Decision: **`AutoModelForSequenceClassification` + LoRA** (head trained fully, LoRA on attention
+ MLP projections), not generating the label as text.

- One forward pass, no decoding → lower and steadier latency.
- A proper softmax over exactly 16 classes → a clean confidence for thresholding, and it can
  never emit an invalid label.
- Rejected alternative: instruction-tune to emit the label, take the label-token probability as
  confidence. More "LLM-like" but slower, multi-token labels make the probability messy, and
  invalid outputs become possible. Record this in DESIGN.md's options-rejected section.

### Config — `config/train_lora.yaml` (starting point)

```yaml
base_model: Qwen/Qwen2.5-1.5B
revision: <pinned commit sha>
max_len: 256
lora: {r: 16, alpha: 32, dropout: 0.05,
       target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]}
lr: 2e-4
epochs: 3
batch_size: 16          # via grad accumulation if needed
warmup_ratio: 0.05
weight_decay: 0.0
bf16: true
gradient_checkpointing: true
seed: 42
quantization: none      # "nf4" for the QLoRA run
```

Small grid on val macro F1: r ∈ {8, 16}, lr ∈ {1e-4, 2e-4}. Expected train time: tens of
minutes per run on the 4060 Ti at ~4.5k short examples — confirm on the first run and record.

### Pad token / head details

Qwen has no default pad token for classification: set `pad_token = eos_token` and
`model.config.pad_token_id`, and ensure the head reads the last non-pad token. Unit-tested,
because this silently ruins accuracy when wrong.

### Inference and measurement — `src/router/lora_model.py`

- `Router.from_pretrained(hub_id_or_path)` → `.predict(text_or_example) -> (intent, conf, probs)`.
- Latency: batch 1, after 20 warm-up calls, CUDA synchronized, on the 4060 Ti. Also measure
  batched throughput (batch 32) for the cost model.
- Optional: CPU-only latency row, since "no GPU" is a realistic deployment for a router.

### Artifacts

- Adapter (`adapter_model.safetensors`, a few tens of MB) + classifier head + `config.json` +
  label map + fitted temperature, pushed to `hf.co/<user>/router-qwen2.5-1.5b-lora` with a model
  card (data, hyperparameters, metrics, limitations). `.gitignore` already excludes weights.
- One-command example in README: `python scripts/route.py "I was charged twice for my order"`.

```
config/train_lora.yaml
src/router/lora_model.py
scripts/train_lora.py, predict.py (extended), route.py, push_to_hub.py
tests/test_lora_model.py (torch-marked; uses a tiny random model, e.g. a hf-internal-testing checkpoint)
```

## Testing plan

- Tiny-model smoke test: one training step on 8 examples; save → load round-trip gives
  identical logits.
- Padding: batched vs one-at-a-time predictions are identical (catches the pad-token bug).
- Label map: saved/loaded label order matches `intents.yaml`.
- `route.py` against a local adapter path returns a valid intent and a confidence in [0, 1].

## Acceptance criteria this PRD unblocks

"Small open model, LoRA/QLoRA fine-tuned"; "training reproducibility: script, exact
hyperparameters, seed, hardware"; "adapter weights or a link, plus a one-command inference
example".

## Risks

- **bitsandbytes on native Windows** has historically been fragile. The bf16 LoRA path doesn't
  need it; if the QLoRA ablation fails natively, run it under WSL2 and record that in the hardware
  notes.
- 8 GB VRAM: if 1.5B bf16 with max_len 256 OOMs, drop max_len to 128 (most messages are short)
  before reaching for quantization.

## Open questions

1. Base model: Qwen2.5-1.5B (recommended), Qwen2.5-0.5B (faster, likely a few points worse —
   worth a second row if time allows), or a Llama-3.2-1B / Gemma-class alternative (licence
   terms are stricter). Pick by licence clarity + fit on 8 GB.
2. Report the QLoRA ablation in the main table or only in DESIGN.md? Recommend DESIGN.md, to keep
   the headline table to the three required approaches plus the hybrid.
