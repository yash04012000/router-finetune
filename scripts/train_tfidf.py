"""Train the TF-IDF + logistic regression baseline and save it.

python -m scripts.train_tfidf
"""

import platform
import time

import sklearn

from router.baselines import tfidf
from router.intents import REPO_ROOT, intent_names
from router.results import save_training_record
from router.splits import load_split


def main() -> None:
    labels = intent_names()
    train_examples, val_examples = load_split("train"), load_split("val")

    start = time.perf_counter()
    model, grid = tfidf.train(train_examples, val_examples, labels)
    wall_seconds = time.perf_counter() - start

    best = max(grid, key=lambda g: g["val_macro_f1"])
    for g in grid:
        print(f"C={g['C']:<4} val macro F1 = {g['val_macro_f1']:.4f}" + ("   <- best" if g is best else ""))

    tfidf.save(model, REPO_ROOT / "models" / "tfidf_lr.joblib")
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
