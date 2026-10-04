"""Build dashboard/dataset.html - a single file you open in a browser to explore the dataset.

    python -m scripts.build_dashboard

No server, no internet needed: the data is embedded in the page. Later PRDs add more pages here
(model comparison, hybrid threshold curve).
"""

import json
from pathlib import Path

import yaml

from router.intents import REPO_ROOT, load_intents
from router.jsonl import read_examples

OUT = REPO_ROOT / "dashboard" / "dataset.html"
SPLITS = ["train", "val", "test"]


def collect() -> dict:
    intents = load_intents()
    groups = yaml.safe_load((REPO_ROOT / "config" / "banking77_groups.yaml").read_text(encoding="utf-8"))[
        "groups"
    ]
    rows = []  # [split, intent, original_label, text] - compact, so the page stays small
    for split in SPLITS:
        for e in read_examples(REPO_ROOT / "data" / f"{split}.jsonl"):
            rows.append([split, e.intent, e.meta.get("original_label", ""), e.text])
    return {
        "splits": SPLITS,
        "intents": [
            {"name": i.name, "description": i.description, "confusable_with": i.confusable_with}
            for i in intents
        ],
        "groups": groups,
        "rows": rows,
    }


def main() -> None:
    template = Path(__file__).with_name("dataset_dashboard.html").read_text(encoding="utf-8")
    payload = json.dumps(collect(), ensure_ascii=False).replace("</", "<\\/")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(template.replace("__DATA__", payload), encoding="utf-8")
    print(f"wrote {OUT}  ({OUT.stat().st_size / 1e6:.1f} MB)  - open it in a browser")


if __name__ == "__main__":
    main()
