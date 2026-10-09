# PRD 3 — Classical baselines: TF-IDF + logistic regression, fine-tuned DistilBERT

Status: 3A TF-IDF done; 3B DistilBERT is split into nine small steps in [03-distilbert/](03-distilbert/00-index.md) · Depends on: PRDs 1-2 · Blocks: PRD 7 (and proves the pipeline for 4-6)

## Summary

Two cheap baselines that set the floor: TF-IDF + logistic regression (trains in seconds on CPU)
and a fine-tuned DistilBERT (the "Transformer baseline" in the resume line). TF-IDF is built
first because it exercises the whole path — load frozen split → predict → write predictions
file → compute metrics — before any money or GPU time is spent.

## Goals

- `tfidf_lr`: scikit-learn pipeline, small hyperparameter search on val, predictions on val
  and test.
- `distilbert`: `distilbert-base-uncased` fine-tuned for sequence classification, early stopping
  on val macro F1, same three prediction sets.
- Both emit calibrated-ish confidences so they can also be evaluated as the "small model" in the
  hybrid (a cheap, interesting extra row in PRD 6).
- Training reproducibility recorded: seed, hyperparameters, hardware, wall time.

## Non-goals

- Heavy tuning. These are baselines; a reasonable, documented config is the goal.

## Design

### Input formatting (shared with PRD 4)

`src/router/format.py::render(example) -> str` returns the message to route. The public data is
single-message, so it is just the text; if an example ever has earlier turns they are prefixed
with role tags (`"[agent] ... [user] ..."`). All non-LLM approaches use this one function.
Truncation to the token budget is done by the tokenizer (`truncation_side="left"`), not here.

### TF-IDF + LR — `src/router/baselines/tfidf.py`

- Word 1-2-grams + char 3-5-grams (char n-grams tolerate typos and word forms: Banking77 has real
  user typos), sublinear TF.
- `LogisticRegression(C ∈ {0.5, 1, 2, 4, 8, 16, 32}, class_weight="balanced")`, chosen on val macro F1
  (the grid was extended past 8 after C=8 turned out best at the edge; C=32 won).
- Confidence: `predict_proba` max. Latency: per-example timing at batch 1 on CPU.
- Model saved with joblib to `models/tfidf_lr.joblib` (7 MB, committed).

### DistilBERT — `src/router/baselines/distilbert.py`

> **Superseded by the step-by-step plan in [03-distilbert/00-index.md](03-distilbert/00-index.md)** (nine small
> learning steps, 3.1 to 3.9). The summary below is the end state those steps build up to.

- HF `Trainer`, max_len 128, lr 5e-5, batch 32, up to 5 epochs, warmup 10%, weight decay 0.01,
  early stopping (patience 1) on val macro F1, seed 42, fp16 on the RTX 4060 Ti.
- Confidence: softmax max, then temperature scaling fit on val (same `calibrate.py` that PRD 4
  uses — built here first).
- Weights not committed (gitignored); pushed to the HF Hub alongside the LoRA adapter in PRD 4.

### Shared runner

`scripts/predict.py --approach tfidf_lr --splits val,test` loads a trained model, runs
inference one example at a time (timing each), and writes `results/predictions/...jsonl` +
`.meta.json` via the PRD 1 writer. A separate `scripts/train_<approach>.py` does training and
writes `results/training/<approach>.json` (hyperparameters, seed, per-epoch val metrics,
hardware string from `torch.cuda.get_device_name`, wall time).

### Calibration — `src/router/calibrate.py`

Single-parameter temperature scaling minimizing NLL on val logits; reports ECE before/after.
Matters because PRD 6 thresholds on confidence: an over-confident model makes every threshold
look worse than it should.

```
src/router/format.py, calibrate.py
src/router/baselines/tfidf.py, distilbert.py
scripts/train_tfidf.py, train_distilbert.py, predict.py
tests/test_format.py, test_tfidf.py, test_calibrate.py, test_distilbert.py (torch-marked)
```

### Playground

`python -m scripts.serve` (see PRD overview). TF-IDF is registered in `src/router/serving.py`;
DistilBERT adds its own loader there when it is trained.

## Testing plan

- `render`: left-truncation keeps the final user turn intact; single-turn examples unchanged.
- TF-IDF: trains on a 50-example toy set and beats chance; save/load gives identical predictions.
- Calibration: temperature on already-calibrated synthetic logits ≈ 1; on over-confident logits
  > 1 and ECE drops.
- DistilBERT (skipped without torch): one training step on 8 examples runs; predictions file
  passes the PRD 1 validator.

## Acceptance criteria this PRD unblocks

"Classical baseline" leg of the three-way comparison; first real rows of the results table.

## Results so far

TF-IDF + LR (C=32): val macro F1 0.936, **test accuracy 0.946, macro F1 0.948**, p50 latency
0.7 ms on CPU. Weakest classes are `transfers` / `transfer_problem` (F1 ~0.91-0.93), the
confusable pair we expected. Well short of 99%, so the dataset is not trivially keyword-solvable
and no changes to PRD 2 are needed.

## Open questions

1. (Resolved) Both baselines are built. TF-IDF is the honesty check on dataset difficulty: if it
   had scored ~99% the dataset would have needed harder confusable examples.
