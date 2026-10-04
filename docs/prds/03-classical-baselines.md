# PRD 3 — Classical baselines: TF-IDF + logistic regression, fine-tuned DistilBERT

Status: Not started · Depends on: PRDs 1-2 · Blocks: PRD 7 (and proves the pipeline for 4-6)

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

### Input formatting (shared with PRD 5)

`src/router/format.py::render(example) -> str` flattens turns into one string:
`"[agent] ... [user] ... [user] <routed message>"`, truncated from the left to the tokenizer's
budget so the routed message is never cut. All non-LLM approaches use this one function.

### TF-IDF + LR — `src/router/baselines/tfidf.py`

- Word 1-2-grams + char 3-5-grams (char n-grams handle the injected typos), sublinear TF.
- `LogisticRegression(C ∈ {0.5, 1, 2, 4, 8}, class_weight="balanced")`, chosen on val macro F1.
- Confidence: `predict_proba` max. Latency: per-example timing at batch 1 on CPU.
- Model saved with joblib to `models/tfidf_lr.joblib` (small enough to commit).

### DistilBERT — `src/router/baselines/distilbert.py`

- HF `Trainer`, max_len 128, lr 5e-5, batch 32, up to 5 epochs, warmup 10%, weight decay 0.01,
  early stopping (patience 1) on val macro F1, seed 42, fp16 on the RTX 4060 Ti.
- Confidence: softmax max, then temperature scaling fit on val (same `calibrate.py` that PRD 5
  uses — built here first).
- Weights not committed (gitignored); pushed to the HF Hub alongside the LoRA adapter in PRD 5.

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

## Testing plan

- `render`: left-truncation keeps the final user turn intact; single-turn examples unchanged.
- TF-IDF: trains on a 50-example toy set and beats chance; save/load gives identical predictions.
- Calibration: temperature on already-calibrated synthetic logits ≈ 1; on over-confident logits
  > 1 and ECE drops.
- DistilBERT (skipped without torch): one training step on 8 examples runs; predictions file
  passes the PRD 1 validator.

## Acceptance criteria this PRD unblocks

"Classical baseline" leg of the three-way comparison; first real rows of the results table.

## Open questions

1. Problem statement says DistilBERT *or* TF-IDF. Recommend both: TF-IDF costs an afternoon and
   shows how much of the task is just keywords — a useful honesty check on dataset difficulty.
   If TF-IDF scores ~99%, the dataset is too easy and PRD 2 needs harder confusable examples
   before going further.
