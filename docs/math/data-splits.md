# Data splits: why train / val / test, leakage, stratifying, and how big is big enough

Code: [`src/router/dataset_build.py`](../../src/router/dataset_build.py),
[`src/router/splits.py`](../../src/router/splits.py) · Tests: `tests/test_dataset_build.py`

## 1. Why three splits

A model's job is to do well on messages it has **never seen**. So we need messages it never saw to
measure that, and we have to be careful about *who* has seen what.

| Split | Used for | Who looks at it |
|---|---|---|
| **train** (9,371) | The model learns from these. Its weights change to fit them. | The model, constantly |
| **val** (1,053) | Making *our* decisions: which learning rate, when to stop training, what confidence threshold to use for fallback. | Us, many times |
| **test** (3,195) | The final score we report. | Us, **once per approach** |

**Why val and test are separate.** Every time we choose something by looking at a score (try
lr = 1e-4 vs 2e-4, pick the better), we slightly tune ourselves to that data. After enough choices the
score on that data is optimistic. If we picked on the test set, our reported number would be
inflated by our own choices. Keeping the test set untouched until the end means the final number is an
honest estimate of future performance. Val absorbs the optimism; test stays clean.

Concretely in this project: the confidence threshold for the hybrid (PRD 6) is chosen on **val** and
its accuracy/cost is then reported on **test**.

## 2. Leakage: the silent way to fool yourself

**Leakage** = information from the test set getting into training. The model then "predicts" things it
has effectively memorised, and the score is meaningless.

Banking77 is built by asking people to write many paraphrases of the same idea, so some messages are
repeated word for word. If "Where is my card?" is in train and also in test, the model gets that test
example right by memory.

What we do about it (`dedup` in `dataset_build.py`):

1. **Normalise** each message (lowercase, strip punctuation, collapse spaces) so "My card?" and "my card"
   count as the same text.
2. **Test wins.** If a text is in test and in train/val, it's removed from train/val. The test set never changes
   because of deduplication.
3. **Conflicts are dropped.** If the same text carries two different intents, the label is unreliable, so
   every copy is dropped. (4 messages dropped.)
4. **Repeats inside a split** keep one copy. (60 messages dropped in total across steps 2 and 4.)

A test in `tests/test_dataset_build.py` re-checks that no normalised text appears in two splits.

**Not covered:** near-duplicates with *different wording* ("my card hasn't come" vs "card still
hasn't arrived") remain. They're legitimate paraphrases of the same intent, which is partly what a
classifier is *supposed* to generalise over, but they make the task a little easier than real traffic.
Listed under limitations.

## 3. Freezing the test set

Splits are written once and their SHA-256 fingerprint is stored in `data/splits.lock.json`. Loading a
split re-computes the fingerprint and refuses to continue if it differs. A SHA-256 hash changes
completely if even one byte changes, so "the test set is identical for every approach" is **checked**
by the loader, not just promised.

## 4. Stratified splitting

We make val by taking 10% of the original Banking77 training data. Plain random sampling can, by bad
luck, give a small intent far fewer than 10%. **Stratified** splitting does the 90/10 split
*separately inside each of the 77 original labels*, so each gets almost exactly 10% in val.

Example: label `X` has 35 training messages. Stratified: `round(35 × 0.1) = 4` go to val, 31 to train.
Always at least a few in val, in proportion.

(We stratify on the 77 original labels, not the 15 groups, so every fine-grained label is
represented in both train and val.)

The original Banking77 **test** set is kept as our test set. Because it is the dataset's standard test set,
our numbers can be compared with published results on Banking77 (with the caveat that we merged
labels into 15 groups, so they are not directly comparable).

## 5. Class imbalance

Counts are uneven (train): `card_payment_problem` 1,362 vs `card_problem` 168, about 8 to 1. This is
because we merged different numbers of original labels (9 vs 3) and Banking77 itself is uneven.

Consequences, and what we do:

- **Accuracy can flatter a model** that does well on the big classes. We therefore always report
  **macro F1** as well (see [metrics.md](metrics.md)), which weighs every intent equally.
- For fine-tuning we may use a class-weighted loss or just report per-class scores; PRD 3 and 4 decide.
- We don't rebalance the test set. It reflects the dataset's natural distribution; per-class results
  show the small intents separately.

## 6. How precise is a score on this test set?

From [metrics.md](metrics.md): the standard error of an accuracy `p` on `n` examples is
`sqrt(p(1-p)/n)`.

| Where | n | at accuracy 0.94 | 95% interval (±1.96·SE) |
|---|---|---|---|
| Whole test set | 3,195 | SE = sqrt(0.94 × 0.06 / 3195) = 0.0042 | **±0.8 points** |
| `refunds` only | 80 | SE = sqrt(0.94 × 0.06 / 80) = 0.0266 | **±5.2 points** |
| `card_problem` only | 120 | SE = sqrt(0.94 × 0.06 / 120) = 0.0217 | **±4.2 points** |

So overall differences below about 1 point between approaches are not meaningful, and a single
intent's score on 80 to 120 test messages is only roughly right. When reporting per-intent results
or confused pairs (PRD 6), small differences are noise. That's why the report shows counts next to
every per-class number.

## 7. How many examples do we need?

The task asked for 5,000+. We have 13,619 in total, so we meet it. For a **LoRA fine-tune of a
pre-trained model**, a few hundred examples per class is typically enough; more helps with
diminishing returns. We have 168 to 1,362 per intent for training, which is a comfortable range
(and the smallest classes are an interesting stress test).
