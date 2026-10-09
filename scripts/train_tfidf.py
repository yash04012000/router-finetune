"""Train the TF-IDF + logistic regression baseline and save it.

python -m scripts.train_tfidf [--verbose]

Progress goes to the console and to logs/router.log.
"""

import argparse
import logging
import platform
import time

import sklearn

from router import log
from router.baselines import tfidf
from router.intents import REPO_ROOT, intent_names
from router.results import save_training_record
from router.splits import load_split

logger = logging.getLogger("router.train_tfidf")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true", help="show DEBUG lines on the console too")
    log.setup_logging(verbose=parser.parse_args().verbose)
    logger.info("starting TF-IDF training (scikit-learn %s)", sklearn.__version__)

    labels = intent_names()
    train_examples, val_examples = load_split("train"), load_split("val")

    start = time.perf_counter()
    model, grid = tfidf.train(train_examples, val_examples, labels)
    wall_seconds = time.perf_counter() - start

    best = max(grid, key=lambda g: g["val_macro_f1"])
    logger.info("training took %.1f s", wall_seconds)

    model_path = REPO_ROOT / "models" / "tfidf_lr.joblib"
    tfidf.save(model, model_path)
    logger.info("saved model to %s (%.1f MB)", model_path, model_path.stat().st_size / 1e6)
    save_training_record(
        tfidf.APPROACH,
        {
            "approach": tfidf.APPROACH,
            "grid": grid,
            "chosen_C": best["C"],
            "n_train": len(train_examples),
            "n_val": len(val_examples),
            "wall_seconds": round(wall_seconds, 1),
            "hardware": f"CPU ({platform.processor() or platform.machine()})",
            "sklearn": sklearn.__version__,
            "seed": "none (LogisticRegression lbfgs is deterministic)",
        },
    )


if __name__ == "__main__":
    main()
