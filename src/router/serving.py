"""One place that knows every model we can run, so the playground UI can show them all.

Each approach is described by a ModelInfo. A model is "available" when its loader can run
(for TF-IDF: the trained file exists). Approaches that come in later PRDs are already listed
with `loader=None`, so the UI shows them greyed out. To plug one in, write a loader that
returns a function `text -> Result` and set it on the entry - nothing else in the UI changes.
"""

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from router.baselines import tfidf
from router.intents import REPO_ROOT

logger = logging.getLogger("router.serving")

TOP_K = 5  # how many runner-up intents to show next to the winner


@dataclass
class Result:
    model: str
    intent: str
    confidence: float | None  # None if the model cannot say how sure it is
    latency_ms: float
    top: list[dict]  # [{"intent": ..., "prob": ...}, ...] best first

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ModelInfo:
    name: str  # matches the `approach` name in predictions files
    label: str
    kind: str  # shown as a tag: classical / fine-tuned / llm / hybrid
    description: str
    planned_in: str  # which PRD adds it
    loader: Callable[[], Callable[[str], Result]] | None = None  # builds the `text -> Result` function
    ready: Callable[[], bool] = lambda: False  # can the loader run right now? (e.g. model file exists)

    model_file: Path | None = None  # shown in the debug page, so you can see what is missing

    @property
    def available(self) -> bool:
        return self.loader is not None and self.ready()

    def why_unavailable(self) -> str:
        """Plain-words reason this model cannot run (empty string if it can)."""
        if self.loader is None:
            return f"not built yet (planned in {self.planned_in})"
        if not self.ready():
            return f"model file missing: {self.model_file} (train it first)"
        return ""


# ---------- TF-IDF ----------
TFIDF_FILE = REPO_ROOT / "models" / "tfidf_lr.joblib"


def load_tfidf() -> Callable[[str], Result]:
    start = time.perf_counter()
    model = tfidf.load(TFIDF_FILE)
    logger.info(
        "loaded tfidf_lr from %s (%.1f MB, %d classes) in %.0f ms",
        TFIDF_FILE, TFIDF_FILE.stat().st_size / 1e6, len(model.classes_), (time.perf_counter() - start) * 1000,
    )  # fmt: skip

    def run(text: str) -> Result:
        start = time.perf_counter()
        probs = model.predict_proba([text])[0]
        latency_ms = (time.perf_counter() - start) * 1000
        order = np.argsort(probs)[::-1][:TOP_K]
        top = [{"intent": str(model.classes_[i]), "prob": float(probs[i])} for i in order]
        return Result("tfidf_lr", top[0]["intent"], top[0]["prob"], latency_ms, top)

    return run


# ---------- the registry ----------
def build_registry() -> dict[str, ModelInfo]:
    models = [
        ModelInfo(
            "tfidf_lr", "TF-IDF + LogReg", "classical",
            "Word and character n-grams into logistic regression. Trains in seconds on a CPU.",
            "PRD 3", load_tfidf, TFIDF_FILE.exists, TFIDF_FILE,
        ),
        ModelInfo(
            "distilbert", "DistilBERT", "fine-tuned",
            "distilbert-base-uncased fine-tuned for 16-way classification.", "PRD 3 (next)",
        ),
        ModelInfo(
            "lora_qwen", "Qwen2.5 + LoRA", "fine-tuned",
            "Small open LLM fine-tuned with LoRA adapters and a classification head.", "PRD 4",
        ),
        ModelInfo(
            "llm_zeroshot", "Large model, zero-shot", "llm",
            "A large model prompted with the intent list. No training.", "PRD 5",
        ),
        ModelInfo(
            "hybrid", "Hybrid router", "hybrid",
            "Small model answers when confident, otherwise falls back to the large model.", "PRD 6",
        ),
    ]  # fmt: skip
    return {m.name: m for m in models}


class Registry:
    """Loads each model the first time it is used, then keeps it in memory."""

    def __init__(self):
        self.infos = build_registry()
        self._loaded: dict[str, Callable[[str], Result]] = {}
        self._lock = threading.Lock()  # the server handles requests in threads; load each model once

    def list_models(self) -> list[dict]:
        return [
            {
                "name": m.name, "label": m.label, "kind": m.kind, "description": m.description,
                "planned_in": m.planned_in, "available": m.available,
            }
            for m in self.infos.values()
        ]  # fmt: skip

    def debug_info(self) -> list[dict]:
        """Everything the debug page shows about each model: status, why not, loaded yet?"""
        return [
            {
                "name": m.name, "available": m.available, "reason": m.why_unavailable(),
                "model_file": str(m.model_file) if m.model_file else None,
                "file_exists": m.model_file.exists() if m.model_file else None,
                "loaded": m.name in self._loaded,
            }
            for m in self.infos.values()
        ]  # fmt: skip

    def predict(self, name: str, text: str) -> Result:
        info = self.infos.get(name)
        if info is None:
            logger.warning("predict asked for unknown model '%s'", name)
            raise KeyError(f"Model '{name}' does not exist")
        if not info.available:
            logger.warning("predict asked for '%s' but it is %s", name, info.why_unavailable())
            raise KeyError(f"Model '{name}' is not available: {info.why_unavailable()}")
        with self._lock:
            if name not in self._loaded:
                logger.info("first use of '%s': loading it now", name)
                self._loaded[name] = info.loader()
        result = self._loaded[name](text)
        logger.debug(
            "%s -> %s (conf %s, %.2f ms) top=%s",
            name, result.intent, None if result.confidence is None else round(result.confidence, 3),
            result.latency_ms, [(t["intent"], round(t["prob"], 3)) for t in result.top[:3]],
        )  # fmt: skip
        return result
