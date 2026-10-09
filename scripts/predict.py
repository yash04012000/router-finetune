"""Run a trained approach over splits, one example at a time, and write the predictions files.

python -m scripts.predict --approach tfidf_lr --splits val,test [--verbose]

Progress goes to the console and to logs/router.log.
"""

import argparse
import logging
import time

from router import log
from router.baselines import tfidf
from router.intents import REPO_ROOT
from router.results import save_predictions
from router.splits import check_predictions_cover, load_split

logger = logging.getLogger("router.predict")


def load_predictor(approach: str):
    """Returns a function: list[Example] -> list[Prediction]. DistilBERT and LoRA get added here."""
    if approach == "tfidf_lr":
        model = tfidf.load(REPO_ROOT / "models" / "tfidf_lr.joblib")
        return lambda examples: tfidf.predict(model, examples)
    logger.error("unknown approach '%s' (known: tfidf_lr)", approach)
    raise SystemExit(f"Unknown approach '{approach}'")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--approach", required=True)
    parser.add_argument("--splits", default="val,test")
    parser.add_argument("--verbose", action="store_true", help="show DEBUG lines on the console too")
    args = parser.parse_args()
    log.setup_logging(verbose=args.verbose)

    predictor = load_predictor(args.approach)
    for split in args.splits.split(","):
        examples = load_split(split)
        logger.info("%s / %s: %d examples", args.approach, split, len(examples))
        start = time.perf_counter()
        preds = predictor(examples)
        logger.info("%s / %s: done in %.1f s", args.approach, split, time.perf_counter() - start)
        check_predictions_cover(preds, examples)  # raises if an example was skipped or answered twice
        path = save_predictions(args.approach, split, preds)
        logger.info("wrote %s", path.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
