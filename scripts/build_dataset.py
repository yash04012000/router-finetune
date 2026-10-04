"""Build data/train.jsonl, val.jsonl, test.jsonl from the raw public data, and freeze them.

    python -m scripts.download_data     # once
    python -m scripts.build_dataset

Running it twice gives byte-identical files (fixed seed, sorted output).
"""

import json

from router.dataset_build import build_splits
from router.intents import REPO_ROOT, intent_names
from router.jsonl import write_examples
from router.splits import write_lock

DATA_DIR = REPO_ROOT / "data"


def main() -> None:
    splits, stats = build_splits(DATA_DIR / "raw", REPO_ROOT / "config" / "banking77_groups.yaml", seed=42)

    # Every example must use an intent from config/intents.yaml, and every intent must be present.
    known = set(intent_names())
    used = {e.intent for exs in splits.values() for e in exs}
    assert used == known, f"intent mismatch: {sorted(used ^ known)}"

    for name, examples in splits.items():
        write_examples(DATA_DIR / f"{name}.jsonl", examples)
    write_lock(DATA_DIR, list(splits))  # fingerprint the files (see router/splits.py)
    (DATA_DIR / "stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    print("counts:", stats["counts"], " dropped:", stats["dropped"])
    print(f"{'intent':28}" + "".join(f"{s:>7}" for s in splits))
    for intent in intent_names():
        print(f"{intent:28}" + "".join(f"{stats['per_intent'][s].get(intent, 0):>7}" for s in splits))


if __name__ == "__main__":
    main()
