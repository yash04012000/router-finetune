"""Load the frozen train / val / test splits.

Why "frozen"? If the test set changes between two experiments, their scores are not comparable.
So when the dataset is created (PRD 2) we record a SHA-256 fingerprint of each split file in
data/splits.lock.json. Loading a split re-computes the fingerprint and refuses to continue if it
doesn't match. A one-character edit to test.jsonl changes the hash completely.
"""

import hashlib
import json
import logging
from pathlib import Path

from router.jsonl import read_examples
from router.schema import Example, Prediction

logger = logging.getLogger("router.splits")

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
LOCK_FILE_NAME = "splits.lock.json"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_lock(data_dir: Path, split_names: list[str]) -> None:
    """Record the fingerprint of each split. Called once, when the dataset is built."""
    data_dir = Path(data_dir)
    lock = {name: file_sha256(data_dir / f"{name}.jsonl") for name in split_names}
    (data_dir / LOCK_FILE_NAME).write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")


def load_split(name: str, data_dir: Path = DATA_DIR) -> list[Example]:
    data_dir = Path(data_dir)
    path = data_dir / f"{name}.jsonl"
    lock = json.loads((data_dir / LOCK_FILE_NAME).read_text(encoding="utf-8"))
    if name not in lock:
        raise KeyError(f"Split '{name}' is not in {LOCK_FILE_NAME}")
    actual = file_sha256(path)
    if actual != lock[name]:
        raise ValueError(
            f"{path} has changed since it was frozen (expected sha256 {lock[name][:12]}..., "
            f"got {actual[:12]}...). Scores would not be comparable with earlier runs."
        )
    logger.debug("split '%s' verified against %s (sha256 %s...)", name, LOCK_FILE_NAME, actual[:12])
    return read_examples(path)


def check_predictions_cover(preds: list[Prediction], examples: list[Example]) -> None:
    """A predictions file must answer every example exactly once - no more, no less."""
    pred_ids = [p.example_id for p in preds]
    if len(pred_ids) != len(set(pred_ids)):
        raise ValueError("Predictions contain duplicate example ids")
    missing = {e.id for e in examples} - set(pred_ids)
    extra = set(pred_ids) - {e.id for e in examples}
    if missing or extra:
        raise ValueError(f"Predictions don't match the split: {len(missing)} missing, {len(extra)} extra")
