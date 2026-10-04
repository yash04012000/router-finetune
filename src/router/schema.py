"""The two record types every part of the project shares.

Example    = one labelled customer message (what we train/test on)
Prediction = one model's answer for one example (what we score)

Every approach (TF-IDF, DistilBERT, LLM, LoRA) writes Predictions in this same shape,
so the scoring code never needs to know which model produced them.
"""

from dataclasses import asdict, dataclass, field

INVALID = "__invalid__"  # what we record when a model answers with something that isn't an intent


@dataclass
class Turn:
    role: str  # "user" or "agent"
    text: str


@dataclass
class Example:
    id: str
    intent: str  # the correct answer (gold label)
    turns: list[Turn]  # the LAST turn is the message being routed; earlier turns are context
    source: str = "synthetic"  # "synthetic" or "handwritten"
    group_id: str = ""  # examples made from the same generation seed share this (see PRD 2)
    meta: dict = field(default_factory=dict)  # tone, length, has_typos, ... (used in error analysis)

    def __post_init__(self):
        if not self.turns:
            raise ValueError(f"Example {self.id} has no turns")

    @property
    def text(self) -> str:
        """The message to route (the last turn)."""
        return self.turns[-1].text

    @staticmethod
    def from_dict(d: dict) -> "Example":
        d = dict(d)
        d["turns"] = [Turn(**t) for t in d["turns"]]
        return Example(**d)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Prediction:
    example_id: str
    approach: str  # e.g. "tfidf_lr", "distilbert", "llm_zeroshot", "lora_qwen"
    predicted: str  # an intent name, or INVALID
    confidence: float | None  # how sure the model is, 0..1 (None if the model can't say)
    latency_ms: float  # time for ONE decision (batch size 1)
    input_tokens: int | None = None  # only API models fill these two in
    output_tokens: int | None = None

    @staticmethod
    def from_dict(d: dict) -> "Prediction":
        return Prediction(**d)

    def to_dict(self) -> dict:
        return asdict(self)
