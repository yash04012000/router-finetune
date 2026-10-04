# Data card

## What this is

13,619 short customer messages, each labelled with one of 16 intents (15 banking intents plus
`out_of_scope`). Used to train and evaluate a support-message router.

| Split | Messages | Source |
|---|---|---|
| `train.jsonl` | 9,371 | Banking77 train (90%) + CLINC150 out-of-scope |
| `val.jsonl` | 1,053 | Banking77 train (10%, stratified) + CLINC150 out-of-scope |
| `test.jsonl` | 3,195 | Banking77 **test** (unchanged apart from deduplication) + CLINC150 out-of-scope |

Per-intent counts: `stats.json`, or open `dashboard/dataset.html` (build it with
`python -m scripts.build_dashboard`). Each file's SHA-256 is in `splits.lock.json`.

## Where the data comes from (no data was generated or paid for)

| Source | What we took | Licence |
|---|---|---|
| **Banking77** (PolyAI; Casanueva et al., 2020, "Efficient Intent Detection with Dual Sentence Encoders") | 13,083 online-banking questions written by crowd workers, 77 intents | CC-BY-4.0 (`licenses/banking77_CC-BY-4.0.txt`) |
| **CLINC150** (Larson et al., 2019, "An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction") | 600 of its 1,200 out-of-scope questions | CC-BY-3.0 (`licenses/clinc150_CC-BY-3.0.txt`) |

The repository's code is MIT; **the data files here remain under the CC-BY licences above** and
require attribution to those authors.

Raw files are downloaded by `python -m scripts.download_data` into `data/raw/` (not committed) and
verified against pinned SHA-256 hashes.

## What we changed

1. **Merged 77 labels into 15 groups** (`config/banking77_groups.yaml`), e.g. `declined_transfer`,
   `failed_transfer`, `pending_transfer` -> `transfer_problem`. Reason: fewer, larger classes are a
   better first fine-tuning problem, and neighbouring groups create realistic confusable pairs.
2. **Sampled 600 CLINC150 out-of-scope questions** (seed 42) as the `out_of_scope` intent, split 70/10/20.
3. **Deduplicated** on normalised text: 60 repeats removed and 4 messages that carried two different
   labels. The test set takes priority, so no test message is also in train or val.
4. **Split** the original Banking77 train set 90/10 per original label (stratified, seed 42).
   The original Banking77 test set is our test set.

Everything is reproducible: `python -m scripts.download_data && python -m scripts.build_dataset`
gives byte-identical files.

## Format

JSON Lines, one example per line:

```json
{"id": "banking77-0019cec4b2", "intent": "top_up", "source": "banking77",
 "turns": [{"role": "user", "text": "Do you have any card fees if I want to add money using an international card?"}],
 "meta": {"original_label": "top_up_by_card_charge"}, "group_id": ""}
```

## Known limitations (read before trusting a score)

- **One domain.** Almost everything is online banking. A model that does well here may not transfer to
  e-commerce or telecoms support.
- **Single-turn.** Every example is one message; there is no conversation history, although the
  schema supports it. Real routing often needs the previous turns.
- **Crowd-written, not production traffic.** People were asked to write queries for a given
  intent. That is cleaner and more on-topic than real customer messages (fewer rants, typos, or
  multi-issue messages).
- **Paraphrase overlap.** Different wordings of the same idea appear in both train and test. We only
  remove identical texts, so scores are somewhat optimistic compared with unseen real traffic.
- **Merged labels are our judgement.** The 77 -> 15 grouping is a choice (e.g. `card_swallowed` went
  to `cash_withdrawal`); a different grouping would give different confusions.
- **Class imbalance.** 168 to 1,362 training examples per intent. Small intents have only 80 to 120
  test examples, so their per-intent scores are imprecise (about +/-4 to 5 points).
- **`out_of_scope` is noisy.** CLINC150's out-of-scope questions are random out-of-domain queries; a
  few look close to banking (e.g. an overdraft-fee question is labelled out of scope there).
- **Single-label.** Each message has exactly one intent; real messages sometimes have two.
