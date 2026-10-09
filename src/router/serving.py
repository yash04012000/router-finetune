"""One place that knows every model we can run, so the playground UI can show them all.

Each approach is described by a ModelInfo. A model is "available" when its loader can run
(for TF-IDF: the trained file exists). Approaches that come in later PRDs are already listed
with `loader=None`, so the UI shows them greyed out. To plug one in, write a loader that
returns a function `text -> Result` and set it on the entry - nothing else in the UI changes.
"""

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np

from router.baselines import tfidf
from router.intents import REPO_ROOT

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

    @property
    def available(self) -> bool:
        return self.loader is not None and self.ready()


# ---------- TF-IDF ----------
TFIDF_FILE = REPO_ROOT / "models" / "tfidf_lr.joblib"


def load_tfidf() -> Callable[[str], Result]:
    model = tfidf.load(TFIDF_FILE)

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
            "PRD 3", load_tfidf, TFIDF_FILE.exists,
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

    def list_models(self) -> list[dict]:
        return [
            {
                "name": m.name, "label": m.label, "kind": m.kind, "description": m.description,
                "planned_in": m.planned_in, "available": m.available,
            }
            for m in self.infos.values()
        ]  # fmt: skip

    def predict(self, name: str, text: str) -> Result:
        info = self.infos.get(name)
        if info is None or not info.available:
            raise KeyError(f"Model '{name}' is not available")
        if name not in self._loaded:
            self._loaded[name] = info.loader()
        return self._loaded[name](text)
