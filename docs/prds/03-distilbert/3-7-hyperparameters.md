# 3.7 — Hyperparameters and overfitting

Status: Not started · Depends on: 3.6 · Next: [3.8](3-8-evaluate-and-compare.md)

## The one idea

**Hyperparameters** are the settings *you* choose (learning rate, epochs, schedule, weight decay), as
opposed to the parameters the model learns. We choose them on the **val** split, with a small, honest
experiment, never by peeking at test. The goal is not the last 0.1%; it is learning which knobs matter and how to read the curves.

## Goal

Run a small, planned set of experiments, compare them in one table, and pick a final configuration.

## Concepts you will learn

- **Learning rate** and why fine-tuning uses a much smaller one (2e-5 to 1e-4) than training from scratch: we are nudging, not rebuilding.
- **Warm-up and decay**: start with a tiny rate, ramp up, then shrink to 0. Why it stabilises early steps and polishes the end.
- **Weight decay**: a gentle pull of weights towards 0 that fights overfitting.
- **Epochs and early stopping**: stop when val macro F1 stops improving (patience), keep the best checkpoint.
- **Variance between runs**: the same config with 3 seeds gives slightly different scores. A 0.2% gap may just be noise (link: the bootstrap and test-size notes in `docs/math/data-splits.md` and `metrics.md`).
- Class imbalance: plain vs class-weighted loss (same idea as `class_weight="balanced"` in TF-IDF).

## What we build

```
scripts/sweep_distilbert.py    runs a short list of configs, one after another, logging each to results/training/
results/training/distilbert_sweep.json   one row per run: config, best val macro F1, best epoch, time
docs/math/fine-tuning.md       part 3: schedules (formula + plot of the rate over steps), weight decay, early stopping
```

Proposed experiments (small on purpose, about 8 runs of a few minutes each):

| Experiment | Values |
|---|---|
| Learning rate | 2e-5, 5e-5, 1e-4 |
| Schedule | constant vs linear warm-up + decay |
| Epochs (with early stopping, patience 1) | up to 5 |
| Loss | plain vs class-weighted |
| Seeds | 3 seeds on the best config, to see the noise |

## Hands-on

1. Run the sweep; read the table together.
2. Plot learning-rate-over-steps for the schedule and train/val curves for the best and worst run.
3. Decide the final config, and write down why in `results/training/distilbert.json`.
4. Retrain the final config once and save the model for 3.8.

## Done when

- One table of all runs, and a chosen config justified by val macro F1 plus the noise estimate.
- The final model is saved.

## UI

Lab view: the experiment table, the learning-rate-over-steps plot for each schedule, and train/val curves of any chosen run side by side.

## Check your understanding

1. Why do we tune on val and not on test?
2. Run A scores 0.951 and run B 0.953 on val. Can you say B is better? What would you check?
3. What does warm-up protect against?
