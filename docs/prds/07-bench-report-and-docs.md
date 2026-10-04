# PRD 7 — One-command bench, report, README, DESIGN.md, limitations

Status: Not started · Depends on: PRDs 1-6 · Blocks: nothing (ship)

## Summary

Turns committed predictions into every number in the README with one command, makes CI prove it,
and writes the docs CONVENTIONS.md requires: README with numbers above the fold, DESIGN.md with
trade-offs and "what changes at 10x", and an honest limitations section.

## Goals

- `python -m scripts.run_bench` (and `make bench` wrapper): reads `results/predictions/*`,
  validates them against the frozen splits, computes all metrics, the hybrid sweep and error
  analysis, writes `results/summary.json`, `results/summary.md`, and all figures. No GPU, no API
  key, runs in seconds.
- `python -m scripts.run_bench --regenerate` documents (and runs, if resources exist) the full
  pipeline: generate data → train → predict → analyze.
- CI runs ruff, pytest, and `run_bench --check`, which fails if the regenerated
  `summary.json` differs from the committed one. That makes "the README numbers are what the code
  produces" a tested property.
- README injection: a script fills the results table between markers
  (`<!-- results:start -->` / `<!-- results:end -->`) from `summary.json`, so the table can't drift.

## Non-goals

- A web dashboard. Static markdown + PNGs.

## Design

### Headline results table (README, first screen)

| Approach | Accuracy (95% CI) | Macro F1 | p95 latency | Cost / 1k decisions | Realism-slice acc |
|---|---|---|---|---|---|
| **Hybrid: LoRA + fallback @ chosen t** | | | | | |
| Large model, zero-shot | | | | | |
| Small model, LoRA | | | | | |
| DistilBERT | | | | | |
| TF-IDF + LR | | | | | |

Hybrid row first ("lead with it"), with the fallback rate in the row label. Directly under the
table: the hybrid curve figure and one sentence with the resume-line numbers.

### README order (per CONVENTIONS.md)

1. One-sentence description. 2. Results table + hybrid curve. 3. Quickstart:
`pip install -r requirements.txt && python -m scripts.run_bench`, then the one-command adapter
inference. 4. Architecture diagram (data gen → splits → three approaches → predictions files →
bench/hybrid). 5. How to reproduce (light path vs full path, with hardware and expected run
times). 6. Limitations.

### Limitations section (must cover)

- Synthetic data: LLM-written, cleaner and more on-label than real traffic; gap shown by the
  realism-slice column. The realism slice is small and written by one person.
- Generator/evaluator overlap mitigated (different families) but not eliminated.
- Single-label assumption; real messages are often multi-intent.
- API latency measured from one location over a short window; prices are snapshots with dates.
- Self-hosted cost depends on the $/GPU-hour assumption and on achieving batched throughput.
- No drift: a fixed taxonomy; adding an intent means retraining, while the prompted model just
  needs a prompt edit — the real operational trade-off.

### DESIGN.md sections

Problem; architecture; options considered and rejected (classification head vs generative labels,
LoRA vs full fine-tune, LiteLLM vs direct SDK, max-prob vs margin confidence, offline hybrid
simulation vs live cascade); failure modes (confident wrong answers bypass fallback, provider
outage during fallback → serve small model's answer and flag, taxonomy drift, out-of-scope
traffic); what changes at 10x (GPU batching/vLLM for the small model, fallback becomes the cost
driver → tune threshold per intent, prompt caching, periodic re-labelling loop from fallback
traffic as new training data).

### Repo hygiene (first push)

GitHub description and 5-6 topics (`lora`, `fine-tuning`, `intent-classification`, `llm-routing`,
`peft`, `cost-optimization`); small real-message commits; no AI attribution trailers.

```
scripts/run_bench.py, update_readme.py
Makefile
results/summary.json, summary.md, figures/
README.md, DESIGN.md
.github/workflows/ci.yml (extended with run_bench --check)
```

## Testing plan

- `run_bench` on a fixture set of tiny predictions files produces a known summary.
- `--check` fails when a committed number is edited by hand.
- README updater is idempotent (running twice changes nothing).

## Acceptance criteria this PRD closes

All five boxes in PROBLEM_STATEMENT.md, verified by walking the list against the shipped repo,
plus the CONVENTIONS.md definition of done: a stranger clones, runs one command, gets the README
numbers.
