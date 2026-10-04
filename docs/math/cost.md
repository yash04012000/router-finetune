# Cost: dollars per 1,000 routing decisions

Code: [`src/router/cost.py`](../../src/router/cost.py), prices in [`config/pricing.yaml`](../../config/pricing.yaml)

To compare a hosted large model with a small model we run ourselves, we need one common unit.
We use **dollars per 1,000 routing decisions**. The two kinds of model are charged differently,
so there are two formulas, plus a third for the hybrid.

## 1. API model: you pay per token

A *token* is a chunk of text (roughly ¾ of an English word). Providers charge separately for
tokens you send (input) and tokens the model writes (output), quoted per million tokens.

```
cost of one decision = input_tokens  · (input price  / 1,000,000)
                     + output_tokens · (output price / 1,000,000)

cost per 1,000       = (average cost of one decision) · 1000
```

**Worked example** (the unit test). Input price $3.00 per million tokens, output $15.00 per million.
Each routing call sends ~1,000 tokens (the 16 intent descriptions plus the customer message) and gets
back 10 tokens (the label):

```
input : 1000 · 3.00  / 1e6 = $0.00300
output:   10 · 15.00 / 1e6 = $0.00015
per decision               = $0.00315      ->  $3.15 per 1,000 decisions
```

Two things to notice:

- The **input dominates**. The intent list is re-sent on every call and is far longer than the answer.
  Provider prompt caching can discount repeated prefix tokens, so we report the plain price as the
  headline and any caching figure separately and labelled.
- We use **real recorded token counts** from each call, not an estimate, and refuse to price a model
  that isn't in `pricing.yaml` rather than silently treating it as free.

## 2. Self-hosted model: you pay per hour

Running our own model on a rented GPU costs money whether the GPU is busy or idle. So the price of
one decision depends on how many decisions the machine completes per hour.

```
decisions per hour = decisions per second · 3600
cost per 1,000     = (GPU price per hour / decisions per hour) · 1000
```

**Worked example** (the unit test). GPU at $0.50/hour, running 100 decisions per second:

```
decisions per hour = 100 · 3600 = 360,000
cost per 1,000     = 0.50 / 360,000 · 1000 = $0.00139
```

That is over 2,000x cheaper than the API example above. The real numbers will differ, but the *shape*
of the argument is this: small model cost is dominated by hardware, and a busy GPU spreads that
cost over a huge number of tiny requests.

### Which throughput do we plug in?

The speed matters hugely, and there are two honest candidates:

| Measure | What it is | Used for |
|---|---|---|
| Batch size 1 | one request at a time, the latency a single user sees | **latency** (p95) |
| Batched (e.g. 32 at once) | what a busy server achieves; the GPU is kept full | **cost** |

Using batch-1 speed for cost would make the small model look many times more expensive than it is
(the GPU would sit mostly idle). We therefore use **batched throughput for cost** and **batch-1 for
latency**, and print both so the assumption is visible.

**Caveat.** The batched figure assumes you have enough traffic to fill batches. At very low volume
you'd pay for a mostly idle GPU. This is stated in the README limitations.

**Training cost** is a one-off and is reported separately, not folded into the per-1,000 figure.

## 3. Hybrid: small model first, large model when unsure

```
cost per 1,000 (hybrid) = small cost + fallback rate · large cost
```

- The small model runs on **every** request, so its full cost is always paid.
- The large model runs only on the **fallback rate**, the fraction of requests whose small-model
  confidence fell below the threshold.

Using the numbers above (small $0.01, large $3.00, per 1,000):

| fallback rate | cost per 1,000 | vs large model alone |
|---|---|---|
| 0% | $0.01 | 0.3% |
| 10% | $0.31 | 10.3% |
| 30% | $0.91 | 30.3% |
| 100% | $3.01 | 100.3% |

At 100% fallback the hybrid is slightly *worse* than the large model alone, because you pay for
both. The whole idea only pays off when the fallback rate is low and the small model's confidence is
a reliable signal of when it's wrong. Measuring exactly that is PRD 6.
