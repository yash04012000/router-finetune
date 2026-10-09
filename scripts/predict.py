"""Run a trained approach over splits, one example at a time, and write the predictions files.

python -m scripts.predict --approach tfidf_lr --splits val,test
"""

import argparse

from router.baselines import tfidf
from router.intents import REPO_ROOT
from router.results import save_predictions
from router.splits import check_predictions_cover, load_split


def load_predictor(approach: str):
    """Returns a function: list[Example] -> list[Prediction]. DistilBERT and LoRA get added here."""
    if approach == "tfidf_lr":
        model = tfidf.load(REPO_ROOT / "models" / "tfidf_lr.joblib")
        return lambda examples: tfidf.predict(model, examples)
    raise SystemExit(f"Unknown approach '{approach}'")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--approach", required=True)
    parser.add_argument("--splits", default="val,test")
    args = parser.parse_args()

    predictor = load_predictor(args.approach)
    for split in args.splits.split(","):
        examples = load_split(split)
        preds = predictor(examples)
        check_predictions_cover(preds, examples)
        path = save_predictions(args.approach, split, preds)
        print(f"wrote {path.relative_to(REPO_ROOT)} ({len(preds)} predictions)")


if __name__ == "__main__":
    main()
