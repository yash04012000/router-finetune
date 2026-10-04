"""Load the intent list from config/intents.yaml.

Everything that needs "the list of possible answers" (the dataset, every model, the metrics)
calls `intent_names()` so there is exactly one source of truth.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
INTENTS_FILE = REPO_ROOT / "config" / "intents.yaml"


@dataclass
class Intent:
    name: str
    description: str
    confusable_with: list[str]


def load_intents(path: Path = INTENTS_FILE) -> list[Intent]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    intents = [Intent(**item) for item in raw["intents"]]

    # Basic sanity checks: a typo here would silently break every later step.
    names = [i.name for i in intents]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate intent names in {path}")
    for intent in intents:
        for other in intent.confusable_with:
            if other not in names:
                raise ValueError(f"{intent.name}: confusable_with has unknown intent '{other}'")
    return intents


def intent_names(path: Path = INTENTS_FILE) -> list[str]:
    """The intents in their fixed order (= confusion-matrix order = class-index order)."""
    return [i.name for i in load_intents(path)]
