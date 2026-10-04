"""Tests for PRD 1: intents, schema, jsonl, splits, metrics, cost."""

import pytest
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix as sk_cm

from router import cost, metrics
from router.intents import intent_names, load_intents
from router.jsonl import read_examples, read_predictions, write_examples, write_predictions
from router.schema import INVALID, Example, Prediction, Turn
from router.splits import check_predictions_cover, load_split, write_lock

LABELS = ["a", "b", "c"]


def make_example(i="x1", intent="a"):
    return Example(id=i, intent=intent, turns=[Turn("user", "hello")])


# ---------- intents ----------
def test_intents_load_and_are_valid():
    names = intent_names()
    assert len(names) == 16
    assert len(set(names)) == 16
    assert "out_of_scope" in names


def test_unknown_confusable_is_rejected(tmp_path):
    f = tmp_path / "i.yaml"
    f.write_text("intents:\n  - {name: a, description: d, confusable_with: [zzz]}\n")
    with pytest.raises(ValueError):
        load_intents(f)


# ---------- schema + jsonl ----------
def test_example_needs_turns():
    with pytest.raises(ValueError):
        Example(id="x", intent="a", turns=[])


def test_examples_round_trip(tmp_path):
    exs = [make_example("1"), make_example("2", "b")]
    write_examples(tmp_path / "e.jsonl", exs)
    assert read_examples(tmp_path / "e.jsonl") == exs


def test_predictions_round_trip(tmp_path):
    preds = [Prediction("1", "m", "a", 0.9, 12.5), Prediction("2", "m", INVALID, None, 3.0, 10, 2)]
    write_predictions(tmp_path / "p.jsonl", preds)
    assert read_predictions(tmp_path / "p.jsonl") == preds


# ---------- splits ----------
def test_split_refuses_edited_file(tmp_path):
    write_examples(tmp_path / "test.jsonl", [make_example()])
    write_lock(tmp_path, ["test"])
    assert len(load_split("test", tmp_path)) == 1
    write_examples(tmp_path / "test.jsonl", [make_example(intent="b")])  # sneaky edit
    with pytest.raises(ValueError, match="changed since it was frozen"):
        load_split("test", tmp_path)


def test_predictions_must_cover_split():
    exs = [make_example("1"), make_example("2")]
    ok = [Prediction("1", "m", "a", 1.0, 1.0), Prediction("2", "m", "a", 1.0, 1.0)]
    check_predictions_cover(ok, exs)
    with pytest.raises(ValueError):
        check_predictions_cover(ok[:1], exs)  # missing one
    with pytest.raises(ValueError):
        check_predictions_cover(ok + ok[:1], exs)  # duplicate


# ---------- metrics ----------
Y_TRUE = ["a", "a", "a", "b", "b", "c", "c", "c", "c", "a"]
Y_PRED = ["a", "a", "b", "b", "c", "c", "c", "a", INVALID, "a"]


def test_accuracy_matches_hand_count():
    # correct at positions 0,1,3,5,6,9 -> 6 / 10
    assert metrics.accuracy(Y_TRUE, Y_PRED) == 0.6


def test_metrics_match_scikit_learn():
    clean_pred = ["a", "a", "b", "b", "c", "c", "c", "a", "b", "a"]  # no INVALID
    assert metrics.accuracy(Y_TRUE, clean_pred) == pytest.approx(accuracy_score(Y_TRUE, clean_pred))
    assert metrics.macro_f1(Y_TRUE, clean_pred, LABELS) == pytest.approx(
        f1_score(Y_TRUE, clean_pred, labels=LABELS, average="macro")
    )
    assert (
        metrics.confusion_matrix(Y_TRUE, clean_pred, LABELS) == sk_cm(Y_TRUE, clean_pred, labels=LABELS)
    ).all()


def test_invalid_counts_as_wrong_and_keeps_label_order():
    cm = metrics.confusion_matrix(Y_TRUE, Y_PRED, LABELS)
    assert cm.shape == (3, 3)
    assert cm.sum() == 9  # the INVALID row is not placed in any column
    assert metrics.invalid_count(Y_PRED) == 1


def test_f1_is_zero_for_class_never_predicted():
    report = metrics.per_class_f1(["a", "b"], ["a", "a"], ["a", "b", "c"])
    assert report["b"]["f1"] == 0.0
    assert report["c"]["f1"] == 0.0  # no examples at all: defined as 0, not a crash


def test_latency_percentiles():
    s = metrics.latency_summary(list(range(1, 101)))  # 1..100 ms
    assert s["p50"] == pytest.approx(50.5)
    assert s["p95"] == pytest.approx(95.05)


def test_bootstrap_ci_is_deterministic_and_brackets_estimate():
    yt = ["a"] * 50 + ["b"] * 50
    yp = ["a"] * 45 + ["b"] * 5 + ["b"] * 48 + ["a"] * 2  # 93% accurate
    lo, hi = metrics.bootstrap_ci(metrics.accuracy, yt, yp, seed=1)
    assert (lo, hi) == metrics.bootstrap_ci(metrics.accuracy, yt, yp, seed=1)
    assert lo < 0.93 < hi


# ---------- cost ----------
PRICING = {"api": {"big": {"input_per_mtok": 3.0, "output_per_mtok": 15.0}}}


def test_api_cost_hand_computed():
    # 1000 in + 10 out per call = 1000*3/1e6 + 10*15/1e6 = 0.00315 per call -> $3.15 per 1k
    preds = [Prediction("1", "llm", "a", None, 100.0, 1000, 10)] * 4
    assert cost.api_cost_per_1k(preds, "big", PRICING) == pytest.approx(3.15)


def test_missing_price_is_an_error_not_free():
    with pytest.raises(KeyError):
        cost.api_cost_per_1k([Prediction("1", "llm", "a", None, 1.0, 1, 1)], "unknown", PRICING)


def test_self_hosted_cost():
    # $0.50/hour, 100 decisions/second = 360,000/hour -> $0.50/360 = $0.001389 per 1k
    assert cost.self_hosted_cost_per_1k(100, 0.50) == pytest.approx(0.50 / 360_000 * 1000)


def test_hybrid_cost_endpoints():
    assert cost.hybrid_cost_per_1k(0.01, 3.0, 0.0) == pytest.approx(0.01)  # never falls back
    assert cost.hybrid_cost_per_1k(0.01, 3.0, 1.0) == pytest.approx(3.01)  # always falls back
    assert cost.hybrid_cost_per_1k(0.01, 3.0, 0.1) == pytest.approx(0.31)
