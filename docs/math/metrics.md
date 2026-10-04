# Metrics: how we score a router

Code: [`src/router/metrics.py`](../../src/router/metrics.py) · Tests: `tests/test_basics.py`

Every approach in this project (TF-IDF, DistilBERT, the large model, LoRA) ends up as the same
thing: a list of true labels `y_true` and a list of predicted labels `y_pred`. Everything below
is computed from those two lists.

## 1. A worked example we'll reuse

10 messages, 3 classes (`a`, `b`, `c`). This is the same data the unit tests use.

| # | true | predicted | | # | true | predicted |
|---|---|---|---|---|---|---|
| 0 | a | a | | 5 | c | c |
| 1 | a | a | | 6 | c | c |
| 2 | a | **b** | | 7 | c | **a** |
| 3 | b | b | | 8 | c | **b** |
| 4 | b | **c** | | 9 | a | a |

## 2. Accuracy

> What fraction of answers were right?

```
accuracy = number correct / total
```

Right at rows 0, 1, 3, 5, 6, 9, so 6 / 10 = **0.60**.

**Limitation.** Accuracy is dominated by common classes. If 95% of messages were `order_status`, a
model that always says `order_status` scores 95% and is useless. That's why we also report macro F1.

**What counts as wrong?** A prediction of `__invalid__` (the LLM answered something that isn't an
intent) is wrong, always. It can never equal a true label, so no special handling is needed.

## 3. Confusion matrix

> For each true class, where did its examples go?

`cm[i][j]` = number of examples whose true class is `i` and predicted class is `j`.
Rows are the truth, columns are the prediction. For the example:

```
              predicted
              a   b   c
   true a  [  3   1   0 ]     <- 4 real a's: 3 right, 1 called b
   true b  [  0   1   1 ]     <- 2 real b's: 1 right, 1 called c
   true c  [  1   1   2 ]     <- 4 real c's: 2 right, 1 called a, 1 called b
```

- The diagonal (3 + 1 + 2 = 6) is the number correct.
- A large off-diagonal cell is a specific confusion. In the real project, a big cell at
  (`refund_request`, `return_exchange`) means refund requests keep being routed to returns.
  PRD 6 reads these cells to find the confused pairs.
- The row and column order is always the order in `config/intents.yaml`, so matrices from
  different models line up cell by cell.

## 4. Precision, recall, F1 (per class)

Look at one class at a time. For class `c` in the example:

- **TP** (true positive): truly `c`, predicted `c` = 2
- **FP** (false positive): truly something else, predicted `c` = 1 (row 4, a `b` called `c`)
- **FN** (false negative): truly `c`, predicted something else = 2 (rows 7 and 8)

```
precision = TP / (TP + FP)   "when I say c, how often am I right?"      = 2 / 3 = 0.667
recall    = TP / (TP + FN)   "of the real c's, how many did I find?"    = 2 / 4 = 0.500
```

In the matrix: **precision = diagonal cell / column total**, **recall = diagonal cell / row total**.

Why both? They fail in opposite ways. A model that says `fraud_security` for every message has
recall 100% (it never misses a fraud case) and terrible precision. A model that says it once,
correctly, has perfect precision and terrible recall. Neither is good.

### F1: one number from the two

```
F1 = 2 · P · R / (P + R)
```

This is the **harmonic mean**. Compare with the plain average when P = 1.0 and R = 0.1:

- arithmetic mean = (1.0 + 0.1) / 2 = 0.55, looks fine
- F1 = 2 · 1.0 · 0.1 / 1.1 = **0.18**, looks as bad as it is

The harmonic mean is pulled towards the smaller value, so you can't get a good F1 by being good at
only one of the two.

For the example: `c` has F1 = 2 · 0.667 · 0.5 / 1.167 = 0.571.

| class | precision | recall | F1 |
|---|---|---|---|
| a | 3/4 = 0.75 | 3/4 = 0.75 | 0.750 |
| b | 1/3 = 0.333 | 1/2 = 0.500 | 0.400 |
| c | 2/3 = 0.667 | 2/4 = 0.500 | 0.571 |

**Edge case.** If a class is never predicted, precision is 0/0. We define that as 0 (and so F1 is
0) instead of crashing. That's the convention scikit-learn uses too.

## 5. Macro F1

```
macro F1 = average of the per-class F1 scores   (each class counts equally)
         = (0.750 + 0.400 + 0.571) / 3 = 0.574
```

Compare: accuracy was 0.60. Macro F1 is lower because class `b` (only 2 examples) did badly, and
macro F1 gives that class the same weight as `a` and `c`. In the real dataset all 16 intents have
roughly the same number of examples, so the gap between accuracy and macro F1 will be small unless
the model is failing on specific intents. A large gap is a signal to look at the per-class table.

**Macro vs micro vs weighted.** *Macro* = unweighted average over classes (what we use). *Micro*
pools all examples first; for single-label classification micro-F1 equals accuracy, so it adds
nothing. *Weighted* averages by class size, so it hides small-class failures like accuracy does.

## 6. Latency percentiles (p50, p95, p99)

> Sort all the request times. The p-th percentile is the value that p% of requests were faster than.

Example: 100 requests taking 1, 2, ..., 100 ms. p50 is 50.5 ms (the median). p95 is 95.05 ms:
`numpy` interpolates between neighbours, the position being `0.95 · (100 - 1) = 94.05` in the sorted
list (0-indexed), i.e. 5% of the way from 95 ms to 96 ms.

**Why p95 and not the average?** A user waits for *their* request, not the average one. A router
that takes 20 ms nineteen times and 2,000 ms once has a mean of ~120 ms but a p99 of 2,000 ms.
Averages hide the tail; the tail is what people complain about. This matters most for the hybrid
in PRD 6: when 10% of requests fall back to a slow API, p50 stays low but p95 jumps.

We measure latency **one request at a time** (batch size 1) because routing happens per turn,
online. Batched speed is only used for the cost model (see [cost.md](cost.md)).

## 7. Bootstrap confidence intervals

> The test set is one random sample of customer messages. A different sample would give a slightly
> different score. How much would it wobble?

**Idea.** Pretend the test set *is* the population. Create a new test set of the same size by
drawing with replacement (some messages appear twice, some not at all), compute the score, and
repeat 1,000 times. Sort the 1,000 scores; the 2.5th and 97.5th percentiles are the **95%
confidence interval**.

```python
for _ in range(1000):
    idx = rng.integers(0, n, size=n)  # n random positions, repeats allowed
    scores.append(metric(y_true[idx], y_pred[idx]))
low, high = percentile(scores, 2.5), percentile(scores, 97.5)
```

**Check against a formula.** For an accuracy `p` measured on `n` examples, the standard error is
`sqrt(p · (1 - p) / n)`. With p = 0.94 and n = 1,280:

```
SE = sqrt(0.94 · 0.06 / 1280) = 0.0066        95% interval ≈ ± 1.96 · SE ≈ ± 1.3 points
```

The bootstrap gives about the same answer, and it also works for macro F1, where no simple formula exists.

**How we use it.** If two approaches differ by less than their intervals, we do **not** claim one is
better ("no clear difference"). With ~1,280 test examples the interval is about ±1.3 points, so a
0.5-point gap is noise.

**What it does not cover.** It only captures sampling noise of *this* test set. It says nothing about
whether the synthetic data resembles real traffic. That is a different risk, handled by the
hand-written realism slice and the limitations section.

**Reproducibility.** The random generator is seeded (`seed=0`), so the same predictions always give
the same interval.
