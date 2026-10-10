"""Tests for step 3.1: the GPU check script. Skipped when torch is not installed (e.g. in CI)."""

import logging

import pytest

torch = pytest.importorskip("torch")

from scripts import check_gpu


def test_matmul_timing_on_cpu_returns_a_positive_time(monkeypatch):
    monkeypatch.setattr(check_gpu, "MATRIX_SIZE", 64)  # tiny, so the test is instant
    assert check_gpu.time_matmul(torch, "cpu", 2) > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA GPU on this machine")
def test_gpu_matmul_matches_cpu_result():
    a, b = torch.randn(64, 64), torch.randn(64, 64)
    on_cpu = a @ b
    on_gpu = (a.cuda() @ b.cuda()).cpu()
    assert torch.allclose(on_cpu, on_gpu, atol=1e-3)  # same maths, tiny float differences allowed


def test_cpu_only_build_gets_a_clear_message(monkeypatch, caplog):
    monkeypatch.setattr(torch.version, "cuda", None)
    # setup_logging() turns propagation off for the "router" logger; caplog needs it on
    monkeypatch.setattr(logging.getLogger("router"), "propagate", True)
    with caplog.at_level("ERROR", logger="router.check_gpu"):
        check_gpu.explain_missing_gpu(torch)
    assert "CPU-only build" in caplog.text
