# Fine-tuning maths

Code: [`src/router/baselines/distilbert.py`](../../src/router/baselines/distilbert.py), [`scripts/explore_model.py`](../../scripts/explore_model.py)
Tests: `tests/test_model_head.py`

This page grows with the steps. Every number here is checked by a test or by the lesson script.

- **Part 1 (step 3.3, below):** what the model computes, from token ids to 16 probabilities. Parameter counts. The loss of a model that guesses.
- Part 2 (step 3.4): cross-entropy over 16 classes in detail, gradients, AdamW, learning rate. *Not written yet.*
- Part 3 (step 3.7): learning-rate schedules, weight decay, early stopping. *Not written yet.*

---

# Part 1: from token ids to 16 probabilities

## 1. The pipeline in one picture

```
text  ->  tokenizer  ->  input_ids [N, L]           N messages, L tokens each (padded)
      ->  body (embeddings + 6 transformer layers)  ->  hidden states [N, L, 768]
      ->  take position 0 (the [CLS] token)         ->  [N, 768]   "summary of the message"
      ->  head: Linear 768->768, ReLU, dropout 0.2, Linear 768->16  ->  logits [N, 16]
      ->  softmax                                   ->  probabilities [N, 16], each row sums to 1
```

- The **body** is pretrained by others on huge amounts of text. It turns every token into a 768-number vector that depends on the *whole* message
  (that is what the transformer layers do, via "attention": each token looks at the others).
- The **head** is new. It is the only part that does not exist in the downloaded file, so it starts with random numbers.
- Why position 0? `[CLS]` is always first, and during pretraining the model learned to use its output as a summary of the sequence.
  We verified in the lesson script that the model's own logits equal `classifier(relu(pre_classifier(hidden[:, 0])))` exactly.

## 2. A Linear layer, with a worked example

`Linear(in -> out)` computes, for an input vector `x` with `in` numbers:

```
z = W x + b          W has out rows and in columns; b has out numbers
parameters = in * out + out
```

Worked example, `Linear(2 -> 3)`, `x = [1, 2]`:

```
W = [[ 0.5, -1.0],      b = [ 0.1,  0.0, -0.2]
     [ 1.0,  0.5],
     [-0.5,  2.0]]

z1 = 0.5*1 + (-1.0)*2 + 0.1 = -1.4
z2 = 1.0*1 +   0.5 *2 + 0.0 =  2.0
z3 = -0.5*1 +  2.0 *2 - 0.2 =  3.3          ->  z = [-1.4, 2.0, 3.3]
```

**ReLU** keeps positive numbers and turns negatives into 0: `ReLU(z) = [0, 2.0, 3.3]`. Without a ReLU between them, two Linear layers collapse into one
(a matrix times a matrix is just another matrix); the ReLU is what lets the head be more than a single Linear layer.

## 3. Our head and how many parameters it has

```
pre_classifier: Linear(768 -> 768)  = 768*768 + 768 = 590,592
classifier:     Linear(768 ->  16)  = 768*16  +  16 =  12,304
                                       head total   = 602,896
```

Counting the whole model (all checked against `model.parameters()`):

| Part | Formula | Parameters |
|---|---|---|
| Word embeddings | 30,522 words x 768 | 23,440,896 |
| Position embeddings | 512 positions x 768 | 393,216 |
| Embedding layer norm | 2 x 768 | 1,536 |
| **Embeddings total** | | **23,835,648** |
| One transformer layer: attention (q, k, v, out) | 4 x (768x768 + 768) | 2,362,368 |
| ... feed-forward (768->3072->768) | (768x3072 + 3072) + (3072x768 + 768) | 4,722,432 |
| ... two layer norms | 2 x 1,536 | 3,072 |
| **One layer total** | | **7,087,872** |
| 6 layers | 6 x 7,087,872 | 42,527,232 |
| **Body total** | 23,835,648 + 42,527,232 | **66,362,880** |
| **Head** | | **602,896** |
| **Whole model** | | **66,965,776** |

The head is **0.90%** of the model. The remaining 99.1% is the pretrained body.

Memory just for the weights, in float32 (4 bytes each): `66,965,776 x 4 = 267.9 MB` (we measured 269 MB on the GPU).
Training needs much more (gradients, optimizer state, activations); step 3.6 does that estimate.

## 4. Logits, softmax, probabilities

(In train mode a dropout with rate 0.2 sits between the ReLU and the last layer; it is switched off in eval mode, so it does not appear in the arithmetic above. Dropout is explained in `docs/learning/step-3-3-model-and-head.md`.)

The head's 16 raw scores are called **logits**. They can be any real number, so they are not yet probabilities. **Softmax** fixes that:

```
p_k = e^(z_k) / (e^(z_1) + ... + e^(z_16))        every p_k > 0, and they sum to 1
```

Worked example with 3 classes, `z = [1.2, -0.3, 0.5]`:

```
e^z  = [3.3201, 0.7408, 1.6487]       sum = 5.7097
p    = [0.5815, 0.1297, 0.2888]       (sums to 1)
```

Properties worth remembering:

- The biggest logit gets the biggest probability, so `argmax(logits) == argmax(probabilities)`. Softmax never changes *which* class wins.
- Adding the same number to every logit changes nothing (it cancels). Only the *differences* between logits matter.
- Equal logits give equal probabilities, `1/16 = 6.25%` for 16 classes.

## 5. Cross-entropy: the loss of a model that guesses

For one message whose correct class is `y`:

```
loss = -ln(p_y)
```

Using the example above: if the true class is index 2, `loss = -ln(0.2888) = 1.2422`. If it were index 0, `loss = -ln(0.5815) = 0.5422`.
Giving the right answer more probability gives a smaller loss. The loss over many messages is the average.

**What does a model that knows nothing score?** It gives every class `1/16`, so `loss = -ln(1/16) = ln(16) = 2.7726`, whatever the true class is.
This is the number every training run *starts* near, and the first thing to check in step 3.4.

Measured on 500 validation messages with our untrained head: **average loss 2.782**, accuracy 7.6% (chance is 6.25%). Slightly above `ln(16)` because the random logits are
not perfectly flat (they have a spread of about 0.1), so the model is a little confidently wrong.

Why is accuracy 7.6% and not exactly 6.25%? The random head happened to predict only 4 of the 16 classes, and `cash_withdrawal` for 58% of messages. That class is
10.7% of validation (113 of 1,053 messages), so always guessing it scores about 10.7% on those messages, and the other few classes dilute that. An untrained model
is not uniformly random per message; it is *consistently* wrong in the same few ways. The number to remember is "near chance, with a loss near `ln(16)`".

## 6. Why the random head gives almost-flat probabilities

Weights in the new head are drawn from a normal distribution with standard deviation 0.02 (BERT's convention; we measured 0.0199 and 0.0200 in the two layers) and the biases start at 0, so logits come out small (we saw values from -0.18 to 0.11).
Small logits mean `e^z` is close to 1 for every class, hence probabilities close to `1/16`. Training will make the logits larger and further apart.
