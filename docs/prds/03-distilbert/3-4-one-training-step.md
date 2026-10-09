# 3.4 — One training step by hand

Status: Not started · Depends on: 3.3 · Next: [3.5](3-5-full-training-loop.md)

## The one idea

Training is a loop of four moves, repeated:

```
1. forward   - run the batch through the model, get logits
2. loss      - measure how wrong the logits are (cross-entropy)
3. backward  - compute, for every parameter, which direction reduces the loss (the gradient)
4. step      - move every parameter a small amount in that direction (optimizer, learning rate)
```

Everything else in fine-tuning (epochs, schedules, LoRA) is bookkeeping around these four moves.

## Goal

Do those four moves by hand, on **one batch of 32 messages**, and watch the loss on that same batch fall
to nearly zero. The test "can the model memorise one batch?" is the standard first sanity check: if it
cannot, something is broken, and we find out before spending hours on a real run.

## Concepts you will learn

- Cross-entropy loss `-ln(p_correct)`, now over 16 classes (you met it in the TF-IDF page).
- What a gradient is, in plain words, and what `loss.backward()` fills in.
- The optimizer **AdamW** and the **learning rate** (how big each move is). Too small: no progress.
  Too big: the loss jumps around or explodes.
- `model.train()` vs `model.eval()` (dropout on or off), and `optimizer.zero_grad()` (why gradients
  must be cleared).
- Why the starting loss is about `ln(16) = 2.77` (random guessing).

## What we build

```
scripts/one_step.py            repeats the four moves ~30 times on a single fixed batch, printing the loss
docs/math/fine-tuning.md       part 2: cross-entropy over 16 classes, gradient descent, AdamW, learning rate
                               (a worked 1-parameter example whose numbers match the code)
```

## Hands-on

1. Print the loss before any step. Check it is near 2.77.
2. Take one step. Print the loss again; it should drop a little.
3. Loop 30 steps; watch it approach 0 (the model memorises the batch).
4. Try learning rates 1e-6, 5e-5, 1e-2. See "too slow", "right", "too big" with your own eyes.
5. Print the size of the gradient for the head vs one body layer to see which parts change most.

## Done when

- Loss falls from about 2.77 to under 0.05 on the fixed batch at lr 5e-5.
- You can explain each of the four moves without looking.

## Check your understanding

1. Why do we call `optimizer.zero_grad()` every step?
2. What goes wrong when the learning rate is far too high?
3. Memorising one batch is not learning the task. Why do we still do this test?
