"""Tests for PRD 2: dataset building (small fixtures) and the real committed splits."""

from collections import Counter

import pytest

from router.dataset_build import dedup, load_group_map, normalize, stratified_split
from router.intents import REPO_ROOT, intent_names
from router.splits import load_split


# ---------- group map ----------
def test_group_map_rejects_unmapped_and_duplicate_labels(tmp_path):
    f = tmp_path / "g.yaml"
    f.write_text("groups:\n  g1: [a, b]\n  g2: [b]\n")
    with pytest.raises(ValueError, match="mapped twice"):
        load_group_map(f, {"a", "b"})
    f.write_text("groups:\n  g1: [a]\n")
    with pytest.raises(ValueError, match="unmapped"):
        load_group_map(f, {"a", "b"})


def test_real_group_map_covers_all_77_labels():
    import csv

    with (REPO_ROOT / "data" / "raw" / "banking77_train.csv").open(encoding="utf-8") as f:
        labels = {row["category"] for row in csv.DictReader(f)}
    if len(labels) != 77:
        pytest.skip("raw data not downloaded")
    mapping = load_group_map(REPO_ROOT / "config" / "banking77_groups.yaml", labels)
    assert len(mapping) == 77
    assert set(mapping.values()) | {"out_of_scope"} == set(intent_names())


# ---------- cleaning ----------
def test_normalize_ignores_case_punctuation_spacing():
    assert normalize("My  card?!") == normalize("my card")


def item(text, intent="x"):
    return {"text": text, "intent": intent}


def test_dedup_test_set_wins_over_train():
    out, dropped = dedup(
        {"test": [item("Where is my card?")], "trainval": [item("where is my card"), item("other")]},
        priority=["test", "trainval"],
    )
    assert [i["text"] for i in out["test"]] == ["Where is my card?"]
    assert [i["text"] for i in out["trainval"]] == ["other"]  # the leaked copy is gone
    assert dropped["duplicate"] == 1


def test_dedup_drops_text_with_two_different_labels():
    out, dropped = dedup(
        {"test": [], "trainval": [item("hello", "a"), item("Hello!", "b"), item("fine", "a")]},
        priority=["test", "trainval"],
    )
    assert [i["text"] for i in out["trainval"]] == ["fine"]
    assert dropped["conflicting_labels"] == 2


# ---------- splitting ----------
def test_stratified_split_keeps_every_label_in_val():
    items = [{"text": str(i), "label": "big"} for i in range(100)] + [
        {"text": f"s{i}", "label": "small"} for i in range(20)
    ]
    train, val = stratified_split(items, 0.1, key="label", seed=0)
    assert Counter(i["label"] for i in val) == {"big": 10, "small": 2}
    assert len(train) + len(val) == 120
    assert not {i["text"] for i in train} & {i["text"] for i in val}


def test_stratified_split_is_deterministic():
    items = [{"text": str(i), "label": "a"} for i in range(50)]
    assert stratified_split(items, 0.2, "label", seed=1) == stratified_split(items, 0.2, "label", seed=1)


# ---------- the real, committed data ----------
@pytest.fixture(scope="module")
def splits():
    try:
        return {name: load_split(name) for name in ("train", "val", "test")}
    except FileNotFoundError:
        pytest.skip("run `python -m scripts.build_dataset` first")


def test_dataset_is_big_enough_and_uses_all_intents(splits):
    assert sum(len(v) for v in splits.values()) >= 5000
    for exs in splits.values():
        assert {e.intent for e in exs} == set(intent_names())


def test_no_text_appears_in_two_splits(splits):
    texts = {name: {normalize(e.text) for e in exs} for name, exs in splits.items()}
    assert not texts["train"] & texts["test"]
    assert not texts["val"] & texts["test"]
    assert not texts["train"] & texts["val"]


def test_ids_are_unique_across_all_splits(splits):
    ids = [e.id for exs in splits.values() for e in exs]
    assert len(ids) == len(set(ids))
