# PRD index — router-finetune

Seven PRDs, built in order over ~2 weeks. Each is buildable and testable before the next starts.
Update the status column as work lands.

| # | PRD | Covers | Est. | Status |
|---|-----|--------|------|--------|
| 1 | [01-foundations.md](01-foundations.md) | Intent taxonomy, shared schemas, predictions-file contract, metrics + cost model, pinned env, CI | 1 day | Done |
| 2 | [02-dataset.md](02-dataset.md) | Public data only (Banking77 merged to 15 intents + CLINC150 out-of-scope), dedup, stratified splits, frozen test set, data card, dataset dashboard | 1 day | Done |
| 3 | [03-classical-baselines.md](03-classical-baselines.md) | TF-IDF + logistic regression, fine-tuned DistilBERT | 1.5 days | Not started |
| 4 | [04-large-model-zero-shot.md](04-large-model-zero-shot.md) | Prompted large model via LiteLLM, structured output, cache/replay, token-based cost | 1.5 days | Not started |
| 5 | [05-small-model-lora.md](05-small-model-lora.md) | LoRA/QLoRA fine-tune of a small open model, calibration, adapter publish, one-command inference | 3 days | Not started |
| 6 | [06-hybrid-and-error-analysis.md](06-hybrid-and-error-analysis.md) | Confidence threshold + fallback sweep, accuracy/cost curve, confused-pair analysis | 1.5 days | Not started |
| 7 | [07-bench-report-and-docs.md](07-bench-report-and-docs.md) | `run_bench` one-command reproduction, results tables/plots, README, DESIGN.md, limitations | 1.5 days | Not started |

## Why this order

PRD 1 fixes the three contracts everything else reads: the intent list, the per-example
predictions file every approach writes (label, confidence, latency, tokens), and the metric/cost
functions that turn those files into the README table. If every approach emits the same
predictions format, the comparison and the hybrid analysis are pure post-processing — no
approach-specific code in the reporting path.

PRD 2 freezes the test set before any model is trained or prompted, so no approach can be tuned
against it. PRDs 3-5 are independent of each other once 1-2 exist; they're ordered cheapest-first
so the pipeline is proven end-to-end on TF-IDF (seconds) before spending API money (PRD 4) or GPU
hours (PRD 5). PRD 4 runs before PRD 5 because the hybrid analysis needs the large model's
predictions on both val and test, and having them early means the fine-tune can be judged against
a real target as it's tuned.

PRD 6 is the headline result ("lead with it") and needs predictions from PRDs 4 and 5. PRD 7 is
integration and proof: it turns committed predictions into every number in the README with one
command and checks the acceptance boxes.

## Problem-statement traceability

| Requirement / acceptance criterion | PRD |
|---|---|
| 5,000+ examples, 10-20 intents, build script published | 2 |
| Large model zero-shot / small LoRA / classical baseline | 4 / 5 / 3 |
| Accuracy, macro F1, p95 latency, cost per 1k, confusion matrix | 1 (functions), 3-5 (inputs), 7 (report) |
| Confused intent pairs; threshold + fallback; accuracy/cost curve | 6 |
| Training script, hyperparameters, seed, hardware | 5 (and 3 for DistilBERT) |
| Adapter weights + one-command inference | 5 |
| Identical held-out test set | 2 (frozen + hashed), 1 (loader enforces hash) |
| Honest limitations incl. dataset realism | 2 (data card limitations), 7 (write-up) |

## Cross-cutting decisions

- **Reproducibility without a GPU or API key.** Every approach writes a predictions JSONL under
  `results/predictions/`, committed to git. `python -m scripts.run_bench` recomputes every README
  number from those files in seconds; CI does exactly this. Regenerating the predictions
  themselves is a separate, documented step that needs the GPU / API key.
- **Thresholds and calibration are fit on val, reported on test.** Never the other way round.
- **Prices are config, not code.** `config/pricing.yaml` holds per-token API prices and the $/GPU-hour
  assumption, each with a source URL and the date it was read.
- **Commits:** no AI attribution trailers (same preference as the rest of this portfolio).
