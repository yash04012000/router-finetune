# Step 3.1: the GPU environment, notes and questions

Plan: [../prds/03-distilbert/3-1-environment.md](../prds/03-distilbert/3-1-environment.md)

## What we did

Found that a **CPU-only** PyTorch was installed, replaced it with the CUDA build, installed `transformers`, and wrote a small script that proves the GPU is used
and shows why we want it.

## Code map

| File | What it does |
|---|---|
| `requirements-train.txt` | Pins `torch==2.14.1+cu130` (CUDA 13.0 build) and `transformers==5.19.0`. Install after `requirements.txt` |
| `scripts/check_gpu.py` | `main()` prints versions, GPU name/VRAM/bf16, a tensor's shape/dtype/device, and times a matrix multiply on CPU vs GPU |
| | `time_matmul(torch, device, repeats)`: warm-up, then timed repeats; `synchronize()` around the clock on the GPU |
| | `explain_missing_gpu(torch)`: says in plain words whether the build is CPU-only or the driver is the problem |
| `tests/test_gpu_env.py` | Skipped when torch is missing; checks GPU and CPU give the same matrix product, and the CPU-only message |

Re-run: `python -m scripts.check_gpu`. Expected last lines: `CPU ... ms per multiply`, `GPU ... ms per multiply -> NNx faster`, `OK: the GPU is ready for training.`

## What we saw

- `nvidia-smi`: RTX 4060 Ti, 8,188 MiB, driver 591.86, supports CUDA up to 13.1.
- `torch 2.14.1+cpu`, built for CUDA `None`, `cuda available: False`. The silent trap.
- After reinstalling `2.14.1+cu130`: GPU found, 8.6 GB, compute capability 8.9, **bf16 supported**.
- 4096x4096 matrix multiply: **393 ms on the CPU, 11 ms on the GPU (36x)**. Peak GPU memory for the check: 210 MB.

## Questions and answers

**1. What happens if you multiply a `cpu` tensor by a `cuda` tensor?**
PyTorch raises an error ("Expected all tensors to be on the same device"). It will not silently copy data between devices, because that copy is slow and
you should choose when it happens. Fix it by moving one tensor: `x.to("cuda")` or `y.to("cpu")`.

**2. Why does the CPU-vs-GPU timing need `torch.cuda.synchronize()` before reading the clock?**
The GPU works asynchronously: the CPU tells it to start and carries on immediately. Reading the clock right after `a @ b` would time only the *launch*
(microseconds), not the multiplication. `synchronize()` makes Python wait until the GPU has finished, so the clock measures the real work. (The CPU
version is synchronous, so it needs nothing.) We also do one **warm-up** multiply first, because the first call pays one-off start-up costs.

**3. Why is `requirements-train.txt` separate from `requirements.txt`?**
CUDA PyTorch is a multi-GB download that needs an NVIDIA GPU. People who only want to run the tests or the scoreboard, and CI, should not need it.
The light file stays fast; the training file is opt-in.

**Why the `+cu130` in the version?** It is a "local version label". PyPI's `torch` is the CPU-only wheel; the CUDA wheels live on PyTorch's own
index (`https://download.pytorch.org/whl/cu130`). Pinning `+cu130` forces pip to pick the CUDA one.

**What does "CUDA 13.1" in `nvidia-smi` mean, if we installed 13.0?** It is the *newest* CUDA your driver can run. A driver that supports 13.1 runs
programs built for 13.0 (and older). The reverse does not work.

**What is bf16, and why do we care that it is supported?** A 16-bit number format with the same range as float32 but less precision. Training in 16-bit uses
half the memory and runs faster; bf16 does not need the "loss scaling" that fp16 needs. Step 3.6 measures this.

## Gotchas

- A CPU-only PyTorch does not complain; code just runs slowly on the CPU. Always check `torch.cuda.is_available()`.
- Use PowerShell for downloads: the Bash tool in this environment has no network.

## Where this connects later

Every training step from 3.3 on starts with `model.to("cuda")`. LoRA (PRD 4) will use the same environment.
