# TF-IDF + logistic regression: the classical baseline

Code: [`src/router/baselines/tfidf.py`](../../src/router/baselines/tfidf.py) · Tests: `tests/test_tfidf.py`

Two steps. TF-IDF turns each message into a vector of numbers. Logistic regression turns that
vector into one probability per intent. Nothing here is a neural network, which is why it trains
in seconds and is a good floor to beat.

## 1. TF-IDF: turning text into numbers

> Give each word a score: high if it is common in *this* message, low if it appears in *every* message.

For a word `w` in a message `d`:

```
tf(w, d)   = 1 + ln(count of w in d)            "sublinear" tf (we use sublinear_tf=True)
idf(w)     = ln((1 + n) / (1 + df(w))) + 1      n = number of messages, df = messages containing w
tfidf(w,d) = tf(w, d) * idf(w)
```

then the whole vector for the message is divided by its length (L2 norm) so long messages don't
beat short ones just by having more words.

- **Why sublinear tf?** A word said 3 times is not 3x as informative. `1 + ln(3) = 2.10`, not 3.
- **Why idf?** "card" appears in many banking messages, so it says little about which intent.
  "pin" appears in few, so it says a lot. Rare words get a bigger idf.
- The `+ 1` and the `1 +` inside the log are scikit-learn's smoothing: they stop a word that
  appears in every message from getting idf 0 and avoid dividing by zero.

### Worked example (verified against scikit-learn)

Three tiny messages: `"card not working"`, `"card payment failed"`, `"top up failed"`. So `n = 3`.

| word | df | idf = ln(4 / (1 + df)) + 1 |
|---|---|---|
| card | 2 | ln(4/3) + 1 = **1.2877** |
| failed | 2 | 1.2877 |
| not, working, payment, top, up | 1 | ln(4/2) + 1 = **1.6931** |

Message 1 (`card not working`), each word once so tf = 1:

```
raw vector   card 1.2877   not 1.6931   working 1.6931
length       sqrt(1.2877^2 + 1.6931^2 + 1.6931^2) = sqrt(7.3913) = 2.7187
normalised   card 0.4736   not 0.6228   working 0.6228
```

The rarer words (`not`, `working`) get more weight than the common one (`card`).

### Three kinds of pieces, and why we use them

We join two vectorizers ("feature union"), so a message's vector has one number for each piece:

- **Words** (1-grams): `card`, `working`.
- **Word pairs** (2-grams): `not working`. A pair captures what single words miss ("not working"
  means something different from "working").
- **Character chunks** (`char_wb`, 3 to 5 letters inside each word, with the word's edges padded
  by spaces). These make the model tolerant to typos and word forms. Example with 3-letter
  chunks: `received` gives `' re', 'rec', 'ece', 'cei', 'eiv', 'ive', 'ved', 'ed '`, and the typo
  `recieved` gives `' re', 'rec', 'eci', 'cie', 'iev', 'eve', 'ved', 'ed '`. They share 4 of 8
  chunks, while as whole words they share nothing. (Chunks of 4 and 5 letters add more overlap.)

## 2. Logistic regression: from numbers to probabilities

Each intent `k` has a weight vector `w_k` (one weight per piece) and a bias `b_k`. For a message
vector `x`:

```
score_k = w_k . x + b_k                     one number per intent (called a logit)
p_k     = exp(score_k) / sum_j exp(score_j)    "softmax": turns scores into probabilities that sum to 1
```

Example with 3 intents and scores `[2, 1, 0]`: `exp` gives `[7.389, 2.718, 1.0]`, sum `11.107`, so
`p = [0.665, 0.245, 0.090]`. The model's answer is the largest, and that largest probability is
the **confidence** that the hybrid router (PRD 6) thresholds on.

### Training: what "learning the weights" means

We choose weights that make the correct intent's probability high on the training messages. The
loss for one message whose true intent is `y` is the **cross-entropy**:

```
loss = -ln(p_y)
```

If the model gave the right answer probability 0.665 the loss is `-ln(0.665) = 0.408`. If it gave
0.09 the loss is 2.41. Confident and right is cheap, confident and wrong is expensive. Training
nudges the weights downhill on the average loss (scikit-learn uses the L-BFGS optimiser; the
idea is the same gradient descent that fine-tuning uses, see `fine-tuning.md` later).

### Regularisation and the C knob

With tens of thousands of pieces and only ~9k messages, the model could memorise the training set.
Regularisation adds a penalty for large weights:

```
total = C * (sum of losses over all training messages) + 0.5 * (sum of all weights squared)
```

`C` is the **inverse** strength of the penalty. Small `C` = strong penalty = simpler model
(might underfit). Large `C` = weak penalty = follows the training data closely (might overfit).
We try `C` in `{0.5, 1, 2, 4, 8, 16, 32}` and keep the one with the best macro F1 on the **val**
split, never the test split (see `data-splits.md`).

### Class weights

`class_weight="balanced"` multiplies each message's loss by `n / (K * n_class)`, where `K` is the
number of intents and `n_class` the number of training messages in that message's class. In this
dataset `card_payment_problem` has 1,362 training messages and `card_problem` has 168, so a
`card_problem` mistake counts about 8x more. Without it, the model could drift toward the big
classes and ignore the small ones, which macro F1 would punish.

## 3. What to expect

TF-IDF + logistic regression only sees which pieces of text are present. It cannot understand
word order or meaning beyond the 2-word pairs. So it does well when an intent has tell-tale
words and struggles when two intents use the same words (for example, `transfers` vs
`transfer_problem`). Those are exactly the weakest classes in its results. Fine-tuned models are
expected to close that gap, and this baseline is how we measure by how much.

Results (val macro F1 chosen `C = 32`; test numbers) are in `results/training/tfidf_lr.json` and
the predictions files; the README table is built from them in PRD 7.
