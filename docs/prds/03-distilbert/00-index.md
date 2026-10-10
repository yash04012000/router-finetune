# PRD 3B — Fine-tuning DistilBERT, in nine small steps

Part of [PRD 3](../03-classical-baselines.md). TF-IDF (PRD 3A) is done. This folder breaks the
DistilBERT half into small steps so each one teaches exactly one idea, and nothing is a black box
by the time we reach the end.

## How we work (the slow-and-clear agreement)

- **One step at a time.** I do not start a step until you say "go". When a step is finished I stop
  and wait, even if the next step looks easy.
- **Explain before, explain after.** Before coding a step I say what we are about to do and why. After it
  I show what happened and what the numbers mean.
- **Small code you can read.** Plain PyTorch, no hidden framework. We write our own training loop
  instead of using Hugging Face's `Trainer`, because the loop *is* the thing to learn. (`Trainer` is
  just this loop with extras, and we will say what it adds.)
- **Look at real things.** Each step has a small script that prints or plots something so you can see the
  idea (token ids, tensor shapes, a loss going down), not just read about it.
- **Check your understanding.** Each step ends with 3 short questions. Answer in your own words if you like;
  I will correct and add. No grades, they are for you.
- **Every step leaves a learning page** in [`docs/learning/`](../../learning/README.md): what we did, a code map, how to re-run,
  what we saw, and the questions **with full answers**, plus any other question you ask along the way. There is also a
  [story so far](../../learning/00-story-so-far.md) and a [glossary](../../learning/glossary.md).
- **Maths goes in `docs/math/`**, with worked numbers checked against code, like the TF-IDF page.
- **Nothing is pushed** until you say so. One small commit per step.
- You can say "slower", "why?", or "skip ahead" at any moment. The plan below is a proposal.

## The nine steps

| Step | Title | The one idea | You will see | Est. |
|---|---|---|---|---|
| [3.1](3-1-environment.md) | Set up the GPU environment | Tensors live on a device (CPU or GPU) | `cuda: True`, a matrix multiply on your RTX 4060 Ti | 1 h |
| [3.2](3-2-tokenizer.md) | Tokenization | A model reads numbers, not words | Your messages as token ids; length histogram | 1-2 h |
| [3.3](3-3-model-and-head.md) | The pretrained model and a fresh head | Pretrained body + new classification head | Shapes, 66M parameters, random-guess accuracy | 2 h |
| [3.4](3-4-one-training-step.md) | One training step by hand | Loss, gradient, optimizer step | The loss on one batch falling to near 0 | 2 h |
| [3.5](3-5-full-training-loop.md) | The full training loop | Batches, epochs, validation | Train vs val curves for 1 to 3 epochs | 3 h |
| [3.6](3-6-speed-and-memory.md) | Fast and fitting in 8 GB | Mixed precision, batch size, memory | Speed and VRAM before vs after | 2 h |
| [3.7](3-7-hyperparameters.md) | Hyperparameters and overfitting | Learning rate, schedule, early stopping | A small experiment table; the best config | 3 h |
| [3.8](3-8-evaluate-and-compare.md) | Evaluate, compare, add to the UI | Same metrics, same test set as TF-IDF | DistilBERT vs TF-IDF in the Scoreboard | 2 h |
| [3.9](3-9-calibration.md) | Calibration | Is "90% sure" right 90% of the time? | Reliability diagram; temperature scaling | 3 h |

Total about 3 days at a relaxed pace. Steps 3.1 to 3.5 are the core; 3.6 to 3.9 make it good and
connect it to the rest of the project. Step 3.9 builds `calibrate.py`, which the LoRA PRD reuses.

## What carries forward to the LoRA PRD (PRD 4)

Almost everything: the tokenizer step, the loop, the speed/memory tricks, and the calibration code
are reused. LoRA changes only *which* parameters train. If you understand 3.1 to 3.9, PRD 4 is a
small step, not a leap.

## Shared rules for every step

- **Data:** only the frozen `train` and `val` splits for training and tuning. The `test` split is
  touched once, in step 3.8, exactly as for TF-IDF.
- **Reproducibility:** seed 42 everywhere; each training run writes a record to
  `results/training/` (hyperparameters, curves, hardware, wall time).
- **Logging:** use the project's logging (see [debugging.md](../../debugging.md)); training prints loss
  every N steps and writes to `logs/router.log`.
- **Hardware:** RTX 4060 Ti 8 GB on Windows 11. DistilBERT (66M parameters) is small enough to train in
  minutes per epoch, so experiments are cheap.
