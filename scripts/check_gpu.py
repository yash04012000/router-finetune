"""Step 3.1: check that PyTorch can see the GPU, and show why we want it.

    python -m scripts.check_gpu

It prints what is installed, shows what a tensor is (shape, dtype, device), then multiplies two big
matrices on the CPU and on the GPU and compares the times. Exit code 1 if the GPU is not usable.
"""

import logging
import platform
import sys
import time

from router import log

logger = logging.getLogger("router.check_gpu")

MATRIX_SIZE = 4096  # two 4096 x 4096 matrices: about 137 billion multiply-adds per product
CPU_REPEATS = 3
GPU_REPEATS = 20


def explain_missing_gpu(torch) -> None:
    """Say, in plain words, why cuda is not available and what to do."""
    if torch.version.cuda is None:
        logger.error(
            "This PyTorch (%s) is a CPU-only build, so it can never use the GPU. "
            "Reinstall the CUDA build listed in requirements-train.txt.",
            torch.__version__,
        )
    else:
        logger.error(
            "PyTorch was built for CUDA %s but cannot reach a GPU. Check `nvidia-smi` works "
            "and that the NVIDIA driver is recent enough for CUDA %s.",
            torch.version.cuda,
            torch.version.cuda,
        )


def time_matmul(torch, device: str, repeats: int) -> float:
    """Average seconds for one (N x N) @ (N x N) product on `device`."""
    a = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device=device)
    b = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device=device)
    a @ b  # warm-up: the first call pays one-off start-up costs we do not want to time
    if device == "cuda":
        # The GPU runs asynchronously: Python moves on before the GPU has finished. Without
        # synchronize() we would stop the clock too early and measure only the launch.
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        a @ b
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeats


def main() -> int:
    log.setup_logging()
    try:
        import torch
    except ImportError:
        logger.error("PyTorch is not installed. Run: pip install -r requirements-train.txt")
        return 1

    logger.info("python %s on %s", sys.version.split()[0], platform.platform())
    logger.info("torch %s (built for CUDA %s)", torch.__version__, torch.version.cuda)
    try:
        import transformers

        logger.info("transformers %s", transformers.__version__)
    except ImportError:
        logger.warning("transformers is not installed (needed from step 3.2)")

    if not torch.cuda.is_available():
        explain_missing_gpu(torch)
        return 1

    props = torch.cuda.get_device_properties(0)
    logger.info(
        "GPU: %s | %.1f GB VRAM | compute capability %d.%d",
        props.name,
        props.total_memory / 1e9,
        props.major,
        props.minor,
    )
    logger.info("bf16 supported: %s", torch.cuda.is_bf16_supported())

    # A tensor is a grid of numbers. These three facts describe every tensor: shape, dtype, device.
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    logger.info("a tensor: shape=%s dtype=%s device=%s", tuple(x.shape), x.dtype, x.device)
    x_gpu = x.to("cuda")  # copies the numbers into GPU memory
    logger.info("after .to('cuda'): device=%s, 2x its values = %s", x_gpu.device, (x_gpu * 2).tolist())

    logger.info(
        "timing a %dx%d matrix multiply (a neural network is mostly many of these)...",
        MATRIX_SIZE,
        MATRIX_SIZE,
    )
    cpu_s = time_matmul(torch, "cpu", CPU_REPEATS)
    gpu_s = time_matmul(torch, "cuda", GPU_REPEATS)
    logger.info("CPU: %.1f ms per multiply", cpu_s * 1000)
    logger.info("GPU: %.1f ms per multiply  ->  %.0fx faster", gpu_s * 1000, cpu_s / gpu_s)
    logger.info("peak GPU memory used by this check: %.0f MB", torch.cuda.max_memory_allocated() / 1e6)
    logger.info("OK: the GPU is ready for training.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
