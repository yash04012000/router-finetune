# The story so far

Written 2026-10-10, updated after step 3.3. Update it as the project moves.

## 1. What we are building, and why

**The task.** A customer sends a message to a bank ("My card was declined at the shop"). A **router** must decide, in one
step, which of 16 **intents** it is (`card_payment_problem`, `top_up`, `transfers`, ..., `out_of_scope`), so it can be sent to the
right team or flow.

**The real goal is learning.** The router is just a concrete problem to learn **fine-tuning** on: take a model that already
knows language and teach it our task. We build several routers and compare them honestly:

| # | Approach | Idea | Status |
|---|---|---|---|
| 3A | TF-IDF + logistic regression | Count words, no neural network. The floor to beat. | Done |
| 3B | DistilBERT fine-tuned | Small pretrained language model + a new head | In progress (step 3.3 of 9 done) |
| 4 | Small LLM (Qwen2.5-1.5B) + LoRA | Same idea, bigger model, train only small adapters | Planned |
| 5 | Large model, zero-shot | Just ask a big model; no training | Planned |
| 6 | Hybrid | Small model answers when confident, else ask the large model | Planned |

The comparison axes: **accuracy, macro F1, latency (p50/p95), and cost per 1,000 decisions.**

## 2. What was built, in order

1. **PRD 1, foundations.** The list of 16 intents (`config/intents.yaml`), the two record types every part shares (`Example`, `Prediction`
   in `src/router/schema.py`), a predictions-file format every model writes, **hand-written metrics** (accuracy, precision/recall/F1, macro F1,
   confusion matrix, percentile latency, bootstrap intervals: `src/router/metrics.py`, explained in `docs/math/metrics.md`), and a cost model.
   *Why hand-written:* so you can see the maths instead of trusting a library.
2. **PRD 2, dataset.** Public data only: **Banking77** (merged from 77 fine-grained intents into 15 broader ones) plus **CLINC150**'s
   out-of-scope questions as the 16th intent. Exact duplicates (60) and messages with conflicting labels (4) were dropped. Stratified split
   into **train 9,371 / val 1,053 / test 3,195**, and the files are **frozen with a SHA-256 hash**: `load_split` refuses to load a file that changed.
   Explained in `docs/math/data-splits.md`. A dataset explorer page lives in `dashboard/` (build with `python -m scripts.build_dashboard`).
3. **PRD 3A, TF-IDF baseline.** Plain scikit-learn. See [tfidf-baseline.md](tfidf-baseline.md).
4. **The playground.** `python -m scripts.serve`: a local web page and JSON API. Type a message, see every available model's answer,
   confidence, latency; a Scoreboard tab scores every model from its committed predictions file; a Debug tab shows logs and request ids.
   New models plug in by one registry entry (`src/router/serving.py`).
5. **Logging.** Every script logs to the console and `logs/router.log` (see [debugging.md](../debugging.md)).
6. **PRD 3B plan.** DistilBERT split into nine small learning steps, one idea each. Steps 3.1 to 3.3 are done:
   [3.1](step-3-1-environment.md), [3.2](step-3-2-tokenizer.md), [3.3](step-3-3-model-and-head.md).
7. **Learning notes.** This folder: story, glossary, a page per step with questions and answers, and a [reading roadmap](reading-roadmap.md) for newcomers.

## 3. Decisions we made, and why

| Decision | Why |
|---|---|
| Public data (Banking77 + CLINC150), no synthetic data | You have no API budget to generate data, and public data is realistic and reproducible |
| Own hand-written metrics and own training loop | The project is for learning; libraries hide the maths |
| Frozen, hashed test set; tune on val only | Otherwise scores from different experiments are not comparable and we would fool ourselves |
| Swapped PRD 4 and 5: small-LLM LoRA before the large zero-shot model | Keeps the learning thread (fine-tune BERT, then fine-tune an LLM), and delays the "which free/local large model" decision |
| Split DistilBERT into nine steps | You asked for the slowest, clearest path; each step teaches one idea |
| Do not `git push` until you say so | Portfolio repo; commits are local and small, with no AI attribution lines |
| PyTorch CUDA build pinned in `requirements-train.txt` | A CPU-only PyTorch was installed and would have trained silently on the CPU (step 3.1) |
| `max_len = 64` | Cuts only 0.29% of training messages (step 3.2) |
| Class index = `intents.yaml` order, stored in the model as `id2label` | One order everywhere (logits, confusion matrix, saved files); a test checks it (step 3.3) |
| Seed 42 set before the new head is created | The head's random start is then the same every run, so experiments are comparable (step 3.3) |
| Check plans against code | The plan said the head was 0.02% of the model; the code showed 0.90%, and we corrected the plan (step 3.3) |

## 4. The numbers so far

| | Val macro F1 | Test accuracy | Test macro F1 | p50 latency |
|---|---|---|---|---|
| TF-IDF + LogReg (C=32) | 0.936 | 0.946 | 0.948 | 0.7 ms (CPU) |
| Random guessing | - | about 0.06 | - | - |
| DistilBERT, **untrained** (random head) | - | 7.6% (val, 500 msgs); loss 2.78 | - | - |

Other facts: the GPU is an RTX 4060 Ti (8 GB); a 4096x4096 matrix multiply takes 393 ms on the CPU and 11 ms on the GPU (36x);
DistilBERT messages average 16 tokens (p95 = 36, p99 = 53, max 98). DistilBERT has 66,965,776 parameters (head 0.90%).

## 5. What is next

Step **3.4**: one training step by hand (forward, loss, backward, optimizer step) on a single batch, watching the loss fall from about 2.78 to near 0.
Then 3.5 (the full loop), 3.6 (speed and memory), and so on. See [the plan](../prds/03-distilbert/00-index.md).
