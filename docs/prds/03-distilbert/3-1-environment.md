# 3.1 — Set up the GPU environment

Status: Not started · Depends on: nothing · Next: [3.2](3-2-tokenizer.md)

## The one idea

A **tensor** is a grid of numbers (a vector, a matrix, a stack of matrices). Neural networks are
tensors being multiplied. A tensor lives on a **device**: the CPU or the GPU. A GPU does the same
multiplications many times faster, but only if the data is moved there first.

## Goal

Install PyTorch with CUDA support and Hugging Face `transformers`, and prove the RTX 4060 Ti is
being used. No model and no data yet.

## Concepts you will learn

- What PyTorch is and what a tensor is (shape, dtype, device).
- What CUDA is, and why the installed PyTorch must be a CUDA build that matches your driver.
- Why we keep training dependencies in a separate `requirements-train.txt` (the light install stays
  fast for CI and for anyone just running the tests).

## What we build

```
requirements-train.txt          torch, transformers, (accelerate not needed - we write our own loop)
scripts/check_gpu.py            prints versions, GPU name, VRAM, runs a timed matrix multiply on CPU vs GPU
tests/test_gpu_env.py           skipped automatically when torch is not installed
```

## Hands-on

1. I install the packages (PowerShell, since the Bash sandbox has no network).
2. Run `python -m scripts.check_gpu` and read the output together:
   `torch 2.x.x`, `cuda available: True`, `NVIDIA GeForce RTX 4060 Ti`, `8.0 GB`.
3. The script multiplies two 4096x4096 matrices on the CPU and on the GPU and prints both times. You
   should see the GPU win by a large factor. That number is the whole reason we train on the GPU.

## Done when

- `cuda available: True` and the GPU name is printed.
- The GPU matrix multiply is clearly faster than the CPU one.
- `pytest` still passes (GPU test skipped or passing), `requirements.txt` untouched.

## Check your understanding

1. What is the difference between a tensor on `cpu` and one on `cuda`? What happens if you multiply
   one of each?
2. Why does the CPU-vs-GPU comparison need `torch.cuda.synchronize()` before reading the clock?
3. Why is `requirements-train.txt` separate from `requirements.txt`?

## Risks

- `torch.cuda.is_available()` is `False`: usually a CPU-only PyTorch was installed, or the NVIDIA driver
  is old. The script prints which case it is and the exact reinstall command to use.
