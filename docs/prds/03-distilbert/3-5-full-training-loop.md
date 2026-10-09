# 3.5 — The full training loop

Status: Not started · Depends on: 3.4 · Next: [3.6](3-6-speed-and-memory.md)

## The one idea

Real training repeats the four moves over the **whole training set many times**. One pass over all
the data is an **epoch**. After each epoch we measure on data the model has *not* trained on (**val**)
to see whether it is learning the task or memorising the examples.

## Goal

A first real training run: our own loop, 1 epoch first, then 3. Save the best model, record the curves.

## Concepts you will learn

- Dataset, batch size, shuffling, `DataLoader`, and why a batch is padded to its longest message (dynamic padding).
- Train loss vs val loss vs val accuracy / macro F1, and what each is for.
- Overfitting: train keeps improving, val stops. How to spot it on a curve.
- Evaluation mode: `torch.no_grad()`, `model.eval()`.
- Checkpoints: saving the best-so-far weights (gitignored; they are about 250 MB).
- Seeds: why two runs with the same seed should match, and what still makes them differ slightly (GPU nondeterminism).

## What we build

```
src/router/baselines/distilbert.py   Dataset class, train_one_epoch, evaluate, train (the loop)
scripts/train_distilbert.py          config at the top (lr, batch size, epochs, max_len, seed), logging, saves
                                     models/distilbert/ and results/training/distilbert.json
tests/test_distilbert.py             tiny-model smoke test (skipped without torch); loop runs on 8 examples
```

Settings to start with (documented, not tuned yet): lr 5e-5, batch 32, max_len from 3.2, AdamW, 1 epoch
then 3, no scheduler yet (that is 3.7).

## Hands-on

1. Run 1 epoch. Read the log: loss every 20 steps, then val accuracy and macro F1 at the end.
2. Compare with TF-IDF's val macro F1 (0.936). Is 1 epoch of DistilBERT already better or worse?
3. Run 3 epochs. Plot or print train loss and val metrics per epoch; find where val stops improving.
4. Run it twice with the same seed and compare the numbers.

## Done when

- A trained model is saved, and `results/training/distilbert.json` holds per-epoch curves, hyperparameters,
  seed, hardware and wall time.
- You can point at the curves and say whether the model underfits, fits or overfits.

## Check your understanding

1. What is the difference between an epoch, a batch, and a step?
2. Train loss keeps falling but val loss rises. What is happening, and what would you do?
3. Why must the val split never be used to update the weights?

## Why our own loop and not `Trainer`?

Hugging Face `Trainer` is this same loop plus logging, schedules, mixed precision and checkpointing.
Writing it ourselves once means you know what each of those options does when you later switch it on.
