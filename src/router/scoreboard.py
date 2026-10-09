"""Score every approach that has a predictions file on the test split, for the UI's Scoreboard tab.

It only reads the committed files in results/predictions/, so it works for any approach
(TF-IDF today; DistilBERT, LoRA, the large model and the hybrid as they land) with no new code.
"""

from router import metrics
from router.intents import REPO_ROOT, intent_names
from router.jsonl import read_predictions
from router.splits import check_predictions_cover, load_split

PREDICTIONS_DIR = REPO_ROOT / "results" / "predictions"
TOP_CONFUSIONS = 5


def top_confusions(y_true: list[str], y_pred: list[str], labels: list[str]) -> list[dict]:
    """The biggest off-diagonal cells of the confusion matrix: 'true X was called Y, n times'."""
    cm = metrics.confusion_matrix(y_true, y_pred, labels)
    cells = [
        (int(cm[i, j]), labels[i], labels[j])
        for i in range(len(labels))
        for j in range(len(labels))
        if i != j
    ]
    cells.sort(reverse=True)
    return [{"true": t, "predicted": p, "count": n} for n, t, p in cells[:TOP_CONFUSIONS] if n > 0]


def score_all(split: str = "test") -> list[dict]:
    labels = intent_names()
    examples = load_split(split)
    y_true = [e.intent for e in examples]
    rows = []
    for path in sorted(PREDICTIONS_DIR.glob(f"*__{split}.jsonl")):
        preds = read_predictions(path)
        check_predictions_cover(preds, examples)
        by_id = {p.example_id: p for p in preds}
        y_pred = [by_id[e.id].predicted for e in examples]
        latency = metrics.latency_summary([by_id[e.id].latency_ms for e in examples])
        rows.append(
            {
                "approach": preds[0].approach,
                "n": len(examples),
                "accuracy": metrics.accuracy(y_true, y_pred),
                "macro_f1": metrics.macro_f1(y_true, y_pred, labels),
                "p50_ms": latency["p50"],
                "p95_ms": latency["p95"],
                "invalid": metrics.invalid_count(y_pred),
                "confusions": top_confusions(y_true, y_pred, labels),
            }
        )
    return rows
