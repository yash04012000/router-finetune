"""Metrics, written out by hand so the maths is visible. See docs/math/metrics.md for the explanations.

Everything starts from two lists of the same length: the true labels and the predicted labels.
"""

import numpy as np

from router.schema import INVALID


def confusion_matrix(y_true: list[str], y_pred: list[str], labels: list[str]) -> np.ndarray:
    """cm[i, j] = how many examples of TRUE class i were predicted as class j.

    Rows = truth, columns = prediction, in the order of `labels` (the intents.yaml order).
    The diagonal is "got it right". An INVALID prediction matches no column, so it only
    shows up as a miss in its true class's row total.
    """
    index = {name: i for i, name in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        if p in index:
            cm[index[t], index[p]] += 1
    return cm


def accuracy(y_true: list[str], y_pred: list[str]) -> float:
    """Fraction of examples answered correctly. INVALID never equals a true label, so it's wrong."""
    return float(np.mean([t == p for t, p in zip(y_true, y_pred)]))


def per_class_f1(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict[str, dict]:
    """Precision, recall and F1 for each class.

    precision = of the times we SAID this class, how often were we right?   TP / (TP + FP)
    recall    = of the times it REALLY was this class, how often did we find it?  TP / (TP + FN)
    F1        = harmonic mean of the two: 2PR / (P + R)  (low if either one is low)
    """
    n = len(y_true)
    true_counts = {c: 0 for c in labels}
    for t in y_true:
        true_counts[t] += 1
    cm = confusion_matrix(y_true, y_pred, labels)
    report = {}
    for i, c in enumerate(labels):
        tp = cm[i, i]
        predicted_as_c = cm[:, i].sum()  # TP + FP  (column total)
        actually_c = true_counts[c]  # TP + FN  (row total, counts INVALID answers as misses)
        precision = tp / predicted_as_c if predicted_as_c else 0.0
        recall = tp / actually_c if actually_c else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        report[c] = {"precision": precision, "recall": recall, "f1": f1, "support": actually_c}
    assert sum(r["support"] for r in report.values()) == n
    return report


def macro_f1(y_true: list[str], y_pred: list[str], labels: list[str]) -> float:
    """Average of per-class F1, every class weighted equally.

    Unlike accuracy, a model can't hide a terrible score on a small class behind good scores on big ones.
    """
    report = per_class_f1(y_true, y_pred, labels)
    return float(np.mean([r["f1"] for r in report.values()]))


def latency_summary(latencies_ms: list[float]) -> dict[str, float]:
    """p50 = the typical request, p95 = the slow tail (95% of requests were faster than this)."""
    a = np.asarray(latencies_ms, dtype=float)
    return {
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "mean": float(a.mean()),
    }


def bootstrap_ci(metric, y_true, y_pred, n_resamples: int = 1000, seed: int = 0, level: float = 0.95):
    """Confidence interval for any metric(y_true, y_pred) by resampling the test set.

    Idea: our test set is just one random sample of possible customer messages. Re-draw
    a test set of the same size WITH replacement, many times, and see how much the score moves.
    The middle 95% of those scores is the interval. See docs/math/metrics.md.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    t = np.asarray(y_true, dtype=object)
    p = np.asarray(y_pred, dtype=object)
    scores = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)  # n random positions, repeats allowed
        scores.append(metric(list(t[idx]), list(p[idx])))
    alpha = (1 - level) / 2
    return float(np.quantile(scores, alpha)), float(np.quantile(scores, 1 - alpha))


def invalid_count(y_pred: list[str]) -> int:
    return sum(p == INVALID for p in y_pred)
