"""Read/write JSONL files (one JSON object per line).

We use this format for the dataset and for predictions: it is plain text, diffs well in git,
and you can open it in any editor.
"""

import json
from pathlib import Path

from router.schema import Example, Prediction


def write_jsonl(path: Path, records: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_examples(path: Path, examples: list[Example]) -> None:
    write_jsonl(path, [e.to_dict() for e in examples])


def read_examples(path: Path) -> list[Example]:
    return [Example.from_dict(d) for d in read_jsonl(path)]


def write_predictions(path: Path, preds: list[Prediction]) -> None:
    write_jsonl(path, [p.to_dict() for p in preds])


def read_predictions(path: Path) -> list[Prediction]:
    return [Prediction.from_dict(d) for d in read_jsonl(path)]
