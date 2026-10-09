# 3.8 — Evaluate, compare with TF-IDF, add to the UI

Status: Not started · Depends on: 3.7 · Next: [3.9](3-9-calibration.md)

## The one idea

A model is only "better" if measured the same way as the others: same frozen test split, same metrics, one
message at a time for latency. This is the first time we open the test set for DistilBERT, once, after all choices are made.

## Goal

Produce DistilBERT's predictions on val and test in the shared predictions format, score them, compare with TF-IDF, and see both
models side by side in the playground.

## Concepts you will learn

- Why latency is measured at batch size 1 (the router decides one message at a time) with warm-up calls first and
  `torch.cuda.synchronize()` around the timer, and why the first call is slow.
- Reading a confusion matrix for a fine-tuned model: which intents does DistilBERT still mix up, and are they the
  same pairs as TF-IDF?
- Confidence = max softmax probability (uncalibrated for now; fixed in 3.9).
- Statistical honesty: the bootstrap confidence interval on test accuracy, to say whether the gap to TF-IDF is real.

## What we build

```
src/router/baselines/distilbert.py   predict() one message at a time with timing, writes Prediction records
scripts/predict.py                   gains --approach distilbert
src/router/serving.py                registers distilbert (loader returns text -> Result); it turns on in the UI
results/predictions/distilbert__{val,test}.jsonl (+ .meta.json)    committed, like TF-IDF
docs/prds/03-classical-baselines.md  results section updated with both models
```

## Hands-on

1. Run `python -m scripts.predict --approach distilbert --splits val,test` and read the log.
2. Open the playground. Select both models; type messages where they might disagree (typos, vague wording,
   two intents in one message). See which one is right more often.
3. Open the Scoreboard: accuracy, macro F1, p50/p95 latency, top confusions for both.
4. Compute the bootstrap 95% interval for the accuracy difference; write down the conclusion in plain words.

## Done when

- `distilbert` shows as available in the playground and the Scoreboard.
- A short written comparison exists: accuracy gain, latency cost (it will be much slower than TF-IDF), and where
  each model fails.

## Check your understanding

1. Why is DistilBERT's latency measured with batch size 1 even though batching is faster?
2. The gap between the models is 2 points on 3,195 test messages. How would you judge whether it is real?
3. Which intents improved most over TF-IDF, and why might a language model help there?
