"""Where predictions and training records are saved, so every approach uses the same layout.

results/predictions/<approach>__<split>.jsonl      the predictions
results/predictions/<approach>__<split>.meta.json  how they were made (hash of the split, versions)
results/training/<approach>.json                   hyperparameters, seed, scores, hardware, wall time
"""

import json
import logging
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from router.intents import REPO_ROOT
from router.jsonl import write_predictions
from router.schema import Prediction
from router.splits import file_sha256

logger = logging.getLogger("router.results")

PREDICTIONS_DIR = REPO_ROOT / "results" / "predictions"
TRAINING_DIR = REPO_ROOT / "results" / "training"


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT, check=False
        )
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def save_predictions(
    approach: str, split: str, preds: list[Prediction], extra_meta: dict | None = None
) -> Path:
    path = PREDICTIONS_DIR / f"{approach}__{split}.jsonl"
    write_predictions(path, preds)
    meta = {
        "approach": approach,
        "split": split,
        "split_sha256": file_sha256(REPO_ROOT / "data" / f"{split}.jsonl"),
        "n": len(preds),
        "git_sha": git_sha(),
        "date": datetime.now(UTC).date().isoformat(),
        "python": sys.version.split()[0],
        "machine": platform.processor() or platform.machine(),
        **(extra_meta or {}),
    }
    path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    logger.info(
        "saved %d predictions to %s (split sha256 %s...)", len(preds), path, meta["split_sha256"][:12]
    )
    return path


def save_training_record(approach: str, record: dict) -> Path:
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    path = TRAINING_DIR / f"{approach}.json"
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    logger.info("saved training record to %s", path)
    return path
