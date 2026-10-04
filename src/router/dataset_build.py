"""Turn the raw public datasets into our train / val / test files.

Read this top to bottom - `build_splits` at the bottom is the whole story:

  1. Map Banking77's 77 labels onto our 15 groups                (load_group_map)
  2. Sample out-of-scope questions from CLINC150                  (build_splits)
  3. Remove duplicate texts, test set wins                        (dedup)
  4. Split Banking77-train into train / val, balanced per label   (stratified_split)

The reasoning (leakage, stratification, sample sizes) is explained in docs/math/data-splits.md.
"""

import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from router.schema import Example, Turn

OUT_OF_SCOPE = "out_of_scope"


# ---------------------------------------------------------------- loading
def load_group_map(path: Path, all_labels: set[str]) -> dict[str, str]:
    """original Banking77 label -> our group name. Every original label must be mapped exactly once."""
    groups = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["groups"]
    mapping: dict[str, str] = {}
    for group, labels in groups.items():
        for label in labels:
            if label in mapping:
                raise ValueError(f"'{label}' is mapped twice ({mapping[label]} and {group})")
            mapping[label] = group
    unmapped = all_labels - set(mapping)
    unknown = set(mapping) - all_labels
    if unmapped or unknown:
        raise ValueError(f"group map mismatch. unmapped: {sorted(unmapped)}  unknown: {sorted(unknown)}")
    return mapping


def read_banking_csv(path: Path) -> list[tuple[str, str]]:
    with Path(path).open(encoding="utf-8") as f:
        return [(row["text"], row["category"]) for row in csv.DictReader(f)]


def read_clinc_oos(path: Path) -> list[str]:
    """CLINC150 stores out-of-scope questions under oos_train / oos_val / oos_test. We pool them."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [text for key in ("oos_train", "oos_val", "oos_test") for text, _label in data[key]]


# ---------------------------------------------------------------- cleaning
def normalize(text: str) -> str:
    """Lowercase, drop punctuation, collapse spaces - so 'My card?' and 'my  card' count as the same text."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", text.lower())).strip()


def dedup(split_items: dict[str, list[dict]], priority: list[str]) -> tuple[dict[str, list[dict]], Counter]:
    """Remove repeated texts. Items are dicts with 'text' and 'intent'.

    Rules, in order:
      - the same text under two different intents is ambiguous -> drop every copy
      - the same text in two splits -> keep it only in the earliest split of `priority`
        (test first, so a test message can never also be a training message = no leakage)
      - the same text twice in one split -> keep one
    """
    intents_by_text = defaultdict(set)
    for items in split_items.values():
        for it in items:
            intents_by_text[normalize(it["text"])].add(it["intent"])
    conflicting = {t for t, ints in intents_by_text.items() if len(ints) > 1}

    dropped = Counter()
    seen: set[str] = set()
    out: dict[str, list[dict]] = {}
    for split in priority:
        kept = []
        for it in split_items[split]:
            key = normalize(it["text"])
            if key in conflicting:
                dropped["conflicting_labels"] += 1
            elif key in seen:
                dropped["duplicate"] += 1
            else:
                seen.add(key)
                kept.append(it)
        out[split] = kept
    return out, dropped


def stratified_split(items: list[dict], val_fraction: float, key: str, seed: int):
    """Split into (train, val) so every value of `key` appears in val in the same proportion.

    Plain random splitting could by bad luck put almost no examples of a small label in val.
    Stratifying = do the split separately inside each label.
    """
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for it in items:
        by_label[it[key]].append(it)
    train, val = [], []
    for label in sorted(by_label):
        group = by_label[label]
        rng.shuffle(group)
        n_val = round(len(group) * val_fraction)
        val += group[:n_val]
        train += group[n_val:]
    return train, val


# ---------------------------------------------------------------- the whole pipeline
def to_example(item: dict) -> Example:
    uid = hashlib.sha1(normalize(item["text"]).encode()).hexdigest()[:10]  # stable id from the text
    return Example(
        id=f"{item['source']}-{uid}",
        intent=item["intent"],
        turns=[Turn("user", item["text"])],
        source=item["source"],
        meta={"original_label": item["original_label"]},
    )


def build_splits(
    raw_dir: Path,
    group_file: Path,
    seed: int = 42,
    val_fraction: float = 0.10,
    n_oos: int = 600,
    oos_fractions: tuple[float, float, float] = (0.70, 0.10, 0.20),
) -> tuple[dict[str, list[Example]], dict]:
    raw_dir = Path(raw_dir)
    train_rows = read_banking_csv(raw_dir / "banking77_train.csv")
    test_rows = read_banking_csv(raw_dir / "banking77_test.csv")
    labels = {label for _, label in train_rows} | {label for _, label in test_rows}
    group_of = load_group_map(group_file, labels)

    def banking_item(text, label):
        return {"text": text, "intent": group_of[label], "source": "banking77", "original_label": label}

    # Banking77's own test set stays our test set, so our numbers are comparable with published ones.
    trainval = [banking_item(t, label) for t, label in train_rows]
    test = [banking_item(t, label) for t, label in test_rows]

    # Out-of-scope: sample n_oos questions from the 1,200 CLINC150 has, then split like the rest.
    rng = random.Random(seed)
    oos_texts = sorted(read_clinc_oos(raw_dir / "clinc150_data_full.json"))
    rng.shuffle(oos_texts)
    oos_texts = oos_texts[:n_oos]
    n_tr, n_va = round(n_oos * oos_fractions[0]), round(n_oos * oos_fractions[1])
    oos = [
        {"text": t, "intent": OUT_OF_SCOPE, "source": "clinc150", "original_label": "oos"} for t in oos_texts
    ]
    oos_train, oos_val, oos_test = oos[:n_tr], oos[n_tr : n_tr + n_va], oos[n_tr + n_va :]

    # Deduplicate BEFORE splitting train/val, so near-identical messages can't end up on both sides.
    cleaned, dropped = dedup(
        {"test": test + oos_test, "trainval": trainval + oos_train + oos_val},
        priority=["test", "trainval"],
    )
    banking_trainval = [it for it in cleaned["trainval"] if it["source"] == "banking77"]
    oos_trainval = [it for it in cleaned["trainval"] if it["source"] == "clinc150"]

    train, val = stratified_split(banking_trainval, val_fraction, key="original_label", seed=seed)
    # out-of-scope was already pre-divided; keep the same train:val ratio after dedup
    n_val_oos = round(len(oos_trainval) * oos_fractions[1] / (oos_fractions[0] + oos_fractions[1]))
    val += oos_trainval[:n_val_oos]
    train += oos_trainval[n_val_oos:]

    splits = {
        name: sorted((to_example(it) for it in items), key=lambda e: e.id)
        for name, items in {"train": train, "val": val, "test": cleaned["test"]}.items()
    }
    stats = {
        "seed": seed,
        "dropped": dict(dropped),
        "counts": {name: len(exs) for name, exs in splits.items()},
        "per_intent": {
            name: dict(sorted(Counter(e.intent for e in exs).items())) for name, exs in splits.items()
        },
    }
    return splits, stats
