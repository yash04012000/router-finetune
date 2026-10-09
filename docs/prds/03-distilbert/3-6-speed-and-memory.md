# 3.6 — Fast, and fitting in 8 GB

Status: Not started · Depends on: 3.5 · Next: [3.7](3-7-hyperparameters.md)

## The one idea

GPU memory (VRAM) and time are the budget. A few cheap tricks decide what you can train: **mixed
precision** (do the maths in 16-bit numbers where it is safe), **batch size**, and **gradient
accumulation** (several small batches that act like one big one). DistilBERT is small enough that
everything fits, so this is the safe place to learn the tricks before LoRA needs them for a 1.5B model.

## Goal

Measure speed and VRAM before and after each trick, so each one is a number, not a belief.

## Concepts you will learn

- Number formats: float32 (4 bytes), float16 / bfloat16 (2 bytes), and why 16-bit halves memory and speeds up the GPU.
- Mixed precision (`torch.autocast`) and loss scaling for fp16; why bf16 does not need it (and whether the RTX 4060 Ti supports bf16).
- What uses VRAM during training: weights, gradients, optimizer state (AdamW keeps 2 extra numbers per parameter), and activations.
  A worked memory estimate for 66M parameters you can check with `torch.cuda.max_memory_allocated`.
- Batch size trade-offs: larger = faster per example but more memory and fewer updates.
- Gradient accumulation: batch 8 x 4 steps behaves like batch 32.
- `max_len` and dynamic padding as free speed-ups.

## What we build

```
scripts/profile_training.py   times 50 steps and records peak VRAM for: fp32, fp16 autocast, bf16 autocast (if
                              supported), batch 16 / 32 / 64, accumulation 4 x 8
docs/math/memory.md           bytes per parameter, optimizer state, the estimate and how it compared
src/router/baselines/distilbert.py   gets an option for the chosen precision and accumulation
```

## Hands-on

1. Run the profiler and read the table of seconds per 50 steps and peak VRAM.
2. Compare your hand estimate of the memory for weights + gradients + AdamW state with what PyTorch reports.
3. Pick the fastest safe setting. Re-run the 3.5 training and confirm the val metrics did not get worse.
4. Optionally force an out-of-memory error with a huge batch and read the error message once, so it is not scary later.

## Done when

- A table of speed and VRAM for each setting is committed, and a default precision is chosen with reasons.
- The 3-epoch run is repeated in the chosen setting with equal quality and shorter time.

## Check your understanding

1. Why does AdamW need more memory than the model weights alone?
2. What does gradient accumulation give you, and what does it not change?
3. Why can fp16 training go wrong (loss turns to NaN) and how does loss scaling help?
