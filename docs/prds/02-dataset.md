# PRD 2 — Dataset: public data, merged intents, dedup, frozen splits

Status: Done · Depends on: PRD 1 · Blocks: PRDs 3-7

> **Change from the original plan:** the first version of this PRD generated synthetic data with an
> LLM plus a hand-written "realism" slice. With no paid API budget, and a goal of learning to
> fine-tune rather than building a perfect benchmark, we switched to **public human-written data
> only**. This also removes the biggest weakness of synthetic data (it flatters every model).

## Summary

13,619 labelled messages across 16 intents, built reproducibly from two public datasets, split so no
message appears in two splits, and frozen by hash before any model sees it.

## Goals (all met)

- 5,000+ examples across 10-20 intents, published in the repo with its build script.
- Real, human-written text. Sources and licences recorded in `data/DATACARD.md`.
- Train / val / test with no leakage; the test set is fixed and fingerprinted.
- A dashboard to look at the data (counts per intent, how labels were merged, a message browser).

## Non-goals

- Generating data with an LLM (no budget; may come back as an optional extra).
- Multi-turn conversations (Banking77 is single-turn; listed as a limitation).

## Design

### 1. Sources

| Source | Used for | Licence |
|---|---|---|
| Banking77 (10,003 train + 3,080 test, 77 intents) | the 15 banking intents | CC-BY-4.0 |
| CLINC150 out-of-scope (1,200 questions, we sample 600) | the `out_of_scope` intent | CC-BY-3.0 |

`scripts/download_data.py` fetches the raw files and verifies pinned SHA-256 hashes.

### 2. Merging 77 labels into 15 intents — `config/banking77_groups.yaml`

Why: 77 classes of ~130 examples each is a harder and noisier first problem; ~15 classes of
hundreds each is a better fine-tuning exercise. Merging *related* labels also produces confusable
neighbours (`transfers` vs `transfer_problem`, `top_up` vs `top_up_problem`, `card_problem` vs
`lost_stolen_security`), which is what PRD 6's error analysis needs to find. The build script
refuses to run unless each of the 77 labels is mapped exactly once.

### 3. Cleaning and splitting — `src/router/dataset_build.py`

1. Normalise text (lowercase, no punctuation) and drop exact repeats; drop texts that carry two
   different intents. The test set has priority, so a test message can never also be in train.
2. Banking77's own test set is our test set (comparable with published work).
3. Banking77 train is split 90/10 into train/val, **stratified** per original label.
4. 600 out-of-scope questions are sampled with a fixed seed and split 70/10/20.
5. Outputs are sorted and seeded, so re-running gives byte-identical files.
6. `splits.lock.json` records each file's SHA-256; `router.splits.load_split` verifies it.

Result: train 9,371 / val 1,053 / test 3,195; 60 repeats and 4 conflicting-label texts removed.

### 4. Explanations

Why three splits, what leakage is, why stratify, and what the test-set size means for confidence
intervals: [docs/math/data-splits.md](../math/data-splits.md).

### Files this PRD produced

```
config/intents.yaml, banking77_groups.yaml
scripts/download_data.py, build_dataset.py, build_dashboard.py, dataset_dashboard.html
src/router/dataset_build.py
data/train.jsonl, val.jsonl, test.jsonl, splits.lock.json, stats.json, DATACARD.md, licenses/
docs/math/data-splits.md
tests/test_dataset_build.py
```

## Testing

- Group map rejects an unmapped or doubly-mapped label; the real map covers all 77 labels.
- Dedup: test wins over train; conflicting labels dropped; punctuation/case ignored.
- Stratified split keeps every label in val; deterministic for a given seed.
- Real committed data: >= 5,000 examples, all 16 intents in every split, no text in two splits,
  unique ids; editing a split file makes `load_split` raise (PRD 1 test).

## Acceptance criteria this PRD satisfies

"Dataset: 5,000+ examples across 10-20 support intents, published with the repo" (source and
processing stated; build script included) and the data half of "honest limitations section covering
dataset realism" (`data/DATACARD.md`).

## Open questions (carried forward)

1. Is the merged-intent grouping sensible? It is our judgement; the dashboard shows exactly which
   original labels sit in each intent, and the grouping can be changed (edit the YAML, rebuild).
2. Optional later extra: add LLM-generated or hand-written messages only if budget/time allows.
