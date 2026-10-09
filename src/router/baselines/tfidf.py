"""Baseline 1: TF-IDF features + logistic regression.

TF-IDF turns a message into a long list of numbers: one per word (or word-pair, or letter-chunk),
big when the piece is frequent in this message but rare across all messages. Logistic regression
then learns one weight per (piece, intent). No neural network, trains in seconds on a CPU.
See docs/math/tfidf-logreg.md for the maths.
"""

import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from router.format import render
from router.metrics import macro_f1
from router.schema import Example, Prediction

APPROACH = "tfidf_lr"
# inverse regularisation strength: bigger C = trust the training data more
C_GRID = [0.5, 1, 2, 4, 8, 16, 32]


def build_pipeline(C: float) -> Pipeline:
    features = FeatureUnion(
        [
            # whole words and word pairs: "card", "not working"
            ("words", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
            # 3-5 letter chunks inside words: "recieved" still shares most chunks with "received"
            ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)),
        ]
    )
    # class_weight="balanced": rare intents count more in the loss, so they are not drowned out.
    classifier = LogisticRegression(C=C, class_weight="balanced", max_iter=1000)
    return Pipeline([("features", features), ("clf", classifier)])


def train(train_examples: list[Example], val_examples: list[Example], labels: list[str]):
    """Fit one model per C on train, keep the one with the best macro F1 on val.

    Returns (best_pipeline, [{"C": ..., "val_macro_f1": ...}, ...]).
    """
    X_train = [render(e) for e in train_examples]
    y_train = [e.intent for e in train_examples]
    X_val = [render(e) for e in val_examples]
    y_val = [e.intent for e in val_examples]

    best, best_score, grid = None, -1.0, []
    for C in C_GRID:
        model = build_pipeline(C).fit(X_train, y_train)
        score = macro_f1(y_val, list(model.predict(X_val)), labels)
        grid.append({"C": C, "val_macro_f1": score})
        if score > best_score:
            best, best_score = model, score
    return best, grid


def save(model: Pipeline, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load(path: Path) -> Pipeline:
    return joblib.load(path)


def predict(model: Pipeline, examples: list[Example]) -> list[Prediction]:
    """One example at a time, timing each (the router decides one message at a time)."""
    preds = []
    for e in examples:
        text = render(e)
        start = time.perf_counter()
        probs = model.predict_proba([text])[0]
        latency_ms = (time.perf_counter() - start) * 1000
        best = int(np.argmax(probs))
        preds.append(
            Prediction(
                example_id=e.id,
                approach=APPROACH,
                predicted=str(model.classes_[best]),
                confidence=float(probs[best]),
                latency_ms=latency_ms,
            )
        )
    return preds
