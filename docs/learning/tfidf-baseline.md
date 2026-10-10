# TF-IDF baseline (PRD 3A): notes and questions

Maths: [../math/tfidf-logreg.md](../math/tfidf-logreg.md) (TF-IDF formula, softmax, cross-entropy, regularisation `C`, class weights, all with checked numbers).

## What we did

Turned each message into a long list of numbers (TF-IDF over words, word pairs, and 3-5 letter chunks) and trained logistic regression on
top. It tries 7 values of `C` and keeps the one with the best **val** macro F1. Predictions are written one message at a time with timing.

## Code map

| File | What it does |
|---|---|
| `src/router/format.py` | `render(example)`: the one place that turns an example into the text a model reads |
| `src/router/baselines/tfidf.py` | `build_pipeline(C)` (the model), `train` (grid over `C`), `save`/`load`, `predict` (one at a time, timed) |
| `src/router/results.py` | Where predictions and training records are written, with a `.meta.json` (split hash, git sha, date) |
| `scripts/train_tfidf.py` | Trains and saves `models/tfidf_lr.joblib` and `results/training/tfidf_lr.json` |
| `scripts/predict.py` | `--approach tfidf_lr --splits val,test` writes `results/predictions/tfidf_lr__{val,test}.jsonl` |
| `tests/test_tfidf.py` | Includes the worked TF-IDF numbers from the maths page, asserted against scikit-learn |

Re-run: `python -m scripts.train_tfidf` then `python -m scripts.predict --approach tfidf_lr --splits val,test` (about 30 s and 5 s).
Re-running rewrites the latency numbers in the predictions files, so do not commit those unless you mean to.

## What we saw

- Val macro F1 rose steadily with `C` (0.911 at 0.5 to 0.936 at 32). The best value was at the edge of the grid twice, so we extended the grid
  from 8 to 32. 47,906 features.
- **Test: accuracy 0.946, macro F1 0.948**, p50 latency 0.7 ms.
- Weakest intents: `transfers` and `transfer_problem` (F1 about 0.91 to 0.93), the pair we guessed would be confusable.

## Questions and answers

**Why is 94.6% a good result for a "dumb" baseline, and why did we not expect 99%?**
Many intents have tell-tale words ("PIN", "exchange rate"), and TF-IDF is good at that. But some intents are separated only by meaning
("where is my transfer?" vs "how do I make a transfer?"), and counting words cannot see meaning or word order beyond 2-word pairs. If it had scored
about 99%, the dataset would have been too easy to tell models apart. 94.6% leaves room for better models to show their value.

**Why use character n-grams as well as words?** They share pieces between related spellings: `received` and `recieved` share 4 of 8
three-letter chunks, but no whole word. See the maths page.

**Why choose `C` on val and not on test?** Choosing on test would make the test score optimistic, because we picked the setting that happens to do best on exactly
those messages. Val is the "practice exam" for choices; test is the "final exam", looked at once.

**Why `class_weight="balanced"`?** `card_payment_problem` has 1,362 training messages and `card_problem` has 168. Without weights, mistakes on the big class
dominate the loss. Balanced weights make a `card_problem` mistake count about 8x more, which is what macro F1 (every class equal) rewards.

**Why is latency measured one message at a time?** The router decides on one message when it arrives. Batching would look faster but would not be what
a user experiences.

**Is training repeatable?** We re-ran it once: all seven validation scores (0.9109 ... 0.9360) matched the first run to four decimals, as expected,
because logistic regression with the lbfgs solver has no randomness. (We did not compare the saved model files byte for byte.) What *does* change on every
re-run is the per-message latency in the predictions files, since timing is never identical. That is why we restore those files after a test run instead
of committing the noise.

## Where this connects later

The **same** prediction format, scoring and playground are used for every later model, so DistilBERT, LoRA, the large model and the hybrid are
compared on identical terms. TF-IDF's probabilities also get calibrated in step 3.9 (the code is model-agnostic).
