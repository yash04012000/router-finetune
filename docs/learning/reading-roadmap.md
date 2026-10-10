# How to read this repo: a roadmap for a beginner

You do not need to read every file, and not in alphabetical order. This page gives you a path. Pick the one that fits your time.

> **Keep this page current.** Whenever a step adds files, add them to the map below (marked with the step).
> Last updated: after step 3.3.

## 0. What this project is, in two sentences

A bank customer sends a message; a **router** decides which of 16 intents it is. We build that router several ways
(word counting, a fine-tuned small language model, a fine-tuned small LLM, a big model with no training, and a hybrid), measure them on the same
held-out messages, and learn how fine-tuning works along the way.

## 1. Three paths

| Path | Time | Do this |
|---|---|---|
| **Tour** | 15 min | Read [00-story-so-far.md](00-story-so-far.md). Run `python -m scripts.serve`, open http://127.0.0.1:8000, type a few messages, open the Scoreboard tab. |
| **Understand the pipeline** | 1-2 h | Follow section 3 below, top to bottom, running each command. |
| **Learn fine-tuning** | days | Do the steps in [the DistilBERT plan](../prds/03-distilbert/00-index.md) one at a time. Each has a lesson script, a learning page with questions and answers, and (from 3.3) a maths page. |

## 2. The map: where things live

```
config/        What the project is about, as data: intents.yaml (the 16 intents), pricing.yaml, banking77_groups.yaml
data/          The dataset: train/val/test .jsonl, splits.lock.json (fingerprints), DATACARD.md (what it is and its limits)
src/router/    The library. Small, readable modules, each with a docstring saying what and why
scripts/       Commands you run: python -m scripts.<name>
tests/         Tests, and also the best examples of how each function is used
models/        Trained model files (TF-IDF is committed; DistilBERT weights are not)
results/       Outputs: predictions/ (every model's answers), training/ (what each run recorded)
ui/            The playground web page
docs/          prds/ = the plan, math/ = the maths, learning/ = this story and the Q&A, debugging.md
logs/          Runtime logs (not in git)
```

## 3. Reading order for the pipeline

Read each file's top docstring first; it is written for you. Run the command, then read the code.

### Layer 1: the ideas everything shares (read first, 20 min)

| Read | Why |
|---|---|
| `PROBLEM_STATEMENT.md` | The original brief: what "done" means |
| `config/intents.yaml` | The 16 answers the router can give. Index `i` in a model's output means the `i`-th intent here, everywhere |
| `src/router/schema.py` | The two record types: `Example` (a message and its right answer) and `Prediction` (a model's answer, with confidence and time) |
| `src/router/intents.py`, `jsonl.py` | Load the intents; read/write one-JSON-per-line files |

### Layer 2: the data (30 min)

| Read / run | Why |
|---|---|
| `data/DATACARD.md` | Where the 13,619 messages came from, and the limitations |
| `docs/math/data-splits.md` | Train vs val vs test, and why the test set is frozen |
| `src/router/splits.py` | `load_split("test")` refuses to load a changed file (SHA-256 check) |
| `src/router/dataset_build.py`, `scripts/build_dataset.py` | How raw public data became the splits (run once; you do not need to re-run it) |
| `python -m scripts.build_dashboard` then open `dashboard/dataset.html` | See the dataset: intents, counts, examples |

### Layer 3: measuring (30 min)

| Read | Why |
|---|---|
| `docs/math/metrics.md` | Accuracy, F1, macro F1, confusion matrix, latency percentiles. Worked numbers |
| `src/router/metrics.py` | The same, hand-written, about 100 lines. Compare each function with the maths page |
| `src/router/cost.py`, `docs/math/cost.md` | Cost per 1,000 decisions (used from PRD 5 on) |

### Layer 4: the first model, TF-IDF + logistic regression (45 min)

| Read / run | Why |
|---|---|
| [tfidf-baseline.md](tfidf-baseline.md), `docs/math/tfidf-logreg.md` | The story and the maths |
| `src/router/format.py` | Turns an example into the text a model reads (one place, so all models see the same input) |
| `src/router/baselines/tfidf.py` | The whole model: about 100 lines |
| `python -m scripts.train_tfidf`, then `python -m scripts.predict --approach tfidf_lr --splits val,test` | Train and write predictions |
| `src/router/results.py` | Where predictions and training records go, and what is recorded with them |

### Layer 5: seeing and debugging (30 min)

| Read / run | Why |
|---|---|
| `python -m scripts.serve` | The playground. Try the three tabs |
| `src/router/serving.py` | **The model registry**: one entry per model. This is how new models appear in the UI |
| `src/router/scoreboard.py` | Scores every predictions file on the test set |
| `scripts/serve.py`, `ui/playground.html` | The server and the page |
| `docs/debugging.md`, `src/router/log.py` | Logs and request ids. Use the Debug tab when something looks wrong |

### Layer 6: fine-tuning DistilBERT (the learning part; one step at a time)

| Step | Files | Page |
|---|---|---|
| 3.1 GPU | `scripts/check_gpu.py`, `requirements-train.txt` | [step-3-1](step-3-1-environment.md) |
| 3.2 Tokenizer | `scripts/explore_tokenizer.py` | [step-3-2](step-3-2-tokenizer.md) |
| 3.3 Model and head | `src/router/baselines/distilbert.py`, `scripts/explore_model.py`, `docs/math/fine-tuning.md` part 1 | [step-3-3](step-3-3-model-and-head.md) |
| 3.4 to 3.9 | *added as we build them* | |

For every step: **read the learning page first** (what and why), **run the lesson script** (see it), then **read the code** (a small file with comments),
then **answer the three questions** before reading the answers.

## 4. How the pieces connect

```
 config/intents.yaml ──────────────┐
                                   v
 public data ─► build_dataset ─► data/{train,val,test}.jsonl   (frozen, hashed)
                                   │
                    load_split() ──┤
                                   v
        ┌── train a model ──► models/…        (tfidf now; DistilBERT, LoRA later)
        │
        └── predict.py ─► results/predictions/<model>__<split>.jsonl  (every model writes the SAME format)
                                   │
              ┌────────────────────┴───────────────────┐
              v                                        v
   scoreboard.py (metrics.py)                  serving.py (registry)
   accuracy, F1, latency, confusions           live answers for a typed message
              └──────────────► ui/playground.html ◄────┘
```

Two rules make this work, and they are worth understanding early:

1. **Every model writes predictions in the same format**, so scoring, the scoreboard and the hybrid never need to know which model produced them.
2. **The test set is frozen and used once**, so every model is judged on identical, untouched data.

## 5. How to read one source file

1. Read the **module docstring**: what the file is for and the idea behind it.
2. Find the **public functions** (no leading underscore). Read only their names and docstrings first.
3. Find the **test** for it in `tests/` (`test_<name>.py`). Tests show real example inputs and outputs.
4. Run the **script** that uses it, with `--verbose` if you want to see more, and read `logs/router.log`.
5. Only then read the function bodies.

## 6. Where to find answers

| Question | Look in |
|---|---|
| What does this word mean? | [glossary.md](glossary.md) |
| Why did we do it this way? | [00-story-so-far.md](00-story-so-far.md), section "Decisions" |
| What is the formula, with numbers? | `docs/math/` |
| Something is broken | `docs/debugging.md`, the Debug tab |
| What is planned next? | `docs/prds/00-overview.md`, `docs/prds/03-distilbert/00-index.md` |
| Why does the test pass or fail? | `tests/`; run `pytest -q -x` |

## 7. What you can safely skip at first

`config/pricing.yaml` and `cost.py` (needed only in PRD 5), `scripts/download_data.py` and `dataset_build.py` (the data is already built),
`DESIGN.md` (a template that PRD 7 fills in), and `docs/prds/04` to `07` until you reach those phases.
