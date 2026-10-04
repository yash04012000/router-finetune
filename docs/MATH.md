# The maths behind this project

Written as we build. Each section says what the formula is, why we use it, and where it lives in the code.

## 1. Accuracy, precision, recall, F1  (`metrics.py`)

Count four things for one class (say `refund_request`):

- TP: truly refund, predicted refund
- FP: truly something else, predicted refund
- FN: truly refund, predicted something else

```
accuracy  = correct / total
precision = TP / (TP + FP)      "when I say refund, am I right?"
recall    = TP / (TP + FN)      "of the real refunds, how many did I catch?"
F1        = 2 * P * R / (P + R) "harmonic mean"
```

**Why harmonic mean?** It is dragged down by the smaller of the two. P=1.0, R=0.1 gives an
arithmetic mean of 0.55 but F1 = 0.18. A classifier that says "refund" only once, correctly, should
not look good.

**Macro F1** = plain average of the 16 per-class F1 scores. Every intent counts equally, so a
model can't hide a bad score on a rare intent behind good scores on common ones. We report it
beside accuracy for that reason.

**Confusion matrix**: `cm[i][j]` = number of true-class-i examples predicted as class j. The diagonal is
correct. A big off-diagonal number is a specific confusion, e.g. true `refund_request` predicted
`return_exchange`.

## 2. Percentile latency  (`metrics.py`)

p95 = the value that 95% of requests were faster than. We report p95 and not the average because
users feel the slow requests, and an average hides them. A router that is 20 ms usually but 2 s
one time in twenty is worse than its mean suggests.

## 3. Bootstrap confidence intervals  (`metrics.py::bootstrap_ci`)

A test accuracy of 94.1% on 1,280 examples is not exact: a different random set of 1,280 customer
messages would score slightly differently. How much?

Bootstrap: pretend the test set is the whole world. Draw 1,280 examples **with replacement**
(some repeat, some missing), score that, repeat 1,000 times. The 2.5th and 97.5th percentiles of
those 1,000 scores form the 95% interval.

Rule of thumb for accuracy near p on n examples: standard error = sqrt(p(1-p)/n).
At p = 0.94, n = 1280: sqrt(0.94 * 0.06 / 1280) = 0.0066, so the 95% interval is about ±1.3 points.
**Consequence:** if two approaches differ by less than about 1-2 points, we say "no clear difference".

## 4. Cost per 1,000 decisions  (`cost.py`)

API model (pay per token):

```
cost per decision = input_tokens * price_in/1e6  +  output_tokens * price_out/1e6
cost per 1k       = average cost per decision * 1000
```

Self-hosted model (pay per hour of machine):

```
cost per 1k = hour_price / (decisions_per_second * 3600) * 1000
```

We use the **batched** throughput, which is what a busy server achieves. Batch-1 speed would
overstate the cost because the GPU would sit mostly idle. Training cost is a one-off and is
reported separately, not added into the per-1k number.

Hybrid (small model first, large model when unsure):

```
cost per 1k = small_cost + fallback_rate * large_cost
```

The small model always runs, the large one only on the fraction of traffic (`fallback_rate`) that
fell below the confidence threshold. This is why the hybrid can be cheap: if only 10% of traffic
falls back, the large-model bill drops by 90%.

(Sections for LoRA, softmax/temperature scaling, calibration error and thresholds are added as
those PRDs are built.)
