# PRD 6 — Hybrid threshold + fallback, and error analysis

Status: Not started · Depends on: PRDs 1, 4, 5 (3 optional) · Blocks: PRD 7

## Summary

The headline finding. Route with the small model; when its confidence is below a threshold,
fall back to the large model. Sweep the threshold to get an accuracy-vs-cost curve, pick an
operating point on val, and report it on test. Alongside it, show exactly where the small model
fails: which intent pairs it confuses, and whether those are the confusions the fallback fixes.

Because every approach already wrote per-example predictions on val and test (PRDs 3-5), the
hybrid is computed offline from those files — no new model calls.

## Goals

- `src/router/hybrid.py`: simulate the hybrid for any (small, fallback, threshold) triple.
- Threshold sweep on val → curve of (fallback rate, accuracy, macro F1, cost/1k, p95 latency).
- Operating points chosen on val by two rules, then evaluated once on test:
  - **Match**: lowest cost whose val accuracy is within 0.5 pt of the large model alone.
  - **Knee**: best accuracy per dollar (or a fixed budget such as ≤ 20% fallback).
- Error analysis: top confused intent pairs for the small model, compared with the designed
  confusable pairs from `intents.yaml`; accuracy split by `meta` (multi-turn vs single,
  tone, length, typos); a few annotated example failures.
- The resume-line numbers ("within X% at Y% of the cost") computed by code, not by hand.

## Non-goals

- Learning a router-of-routers or a cascade with more than two stages (DESIGN.md "at 10x" note).

## Design

### Simulation

For each example: `use_fallback = small.confidence < t`. Final prediction = fallback's if used,
else small's.

- Cost/1k = small cost/1k (always runs) + fallback rate × large cost/1k.
- Latency per example = small latency + (large latency if fallback). p95 over that mixture —
  report it honestly, because the fallback tail dominates p95 once the fallback rate exceeds ~5%.
  Also report p50, which is where the hybrid looks best.
- Thresholds: every distinct confidence value on val (exact curve), plotted with fallback rate on
  x and accuracy on y, cost on a secondary axis.

Also run the same simulation with DistilBERT as the "small" model — if DistilBERT + fallback
matches the LoRA hybrid, that's an important (and honest) finding about whether the LLM-sized
small model is worth it.

### Statistical care

- Operating point picked on val, reported on test: no peeking.
- Bootstrap CIs (PRD 1) on the test accuracy of the chosen point and of the large model alone;
  "matches" is claimed only if CIs overlap, otherwise the gap is stated.
- Oracle-fallback line (fallback exactly on the small model's errors) on the same plot as an
  upper bound, so readers see how good the confidence signal is versus perfect.
- Selective-classification view: accuracy of the small model on the examples it *keeps*, by
  coverage (risk-coverage curve). Same data, different lens — cheap to add.

### Outputs

```
src/router/hybrid.py, errors.py
scripts/analyze_hybrid.py, analyze_errors.py
results/hybrid/sweep_val.csv, sweep_test.csv, operating_points.json
results/errors/confused_pairs.csv, slices.csv, examples.md
results/figures/hybrid_curve.png, confusion_<approach>.png, risk_coverage.png
tests/test_hybrid.py, test_errors.py
```

### Playground

Register this approach in `src/router/serving.py` (loader returning `text -> Result`) so it shows
up in `python -m scripts.serve` next to the other models and in the Scoreboard tab.

## Testing plan

- Threshold 0 → identical to small model; threshold > 1 → identical to large model (all
  fallback); cost and fallback rate at both ends match hand-computed values.
- Monotonicity: fallback rate is non-decreasing in t.
- Operating-point selection uses only val rows (test file deleted in a fixture → still works).
- Confused-pairs on a hand-built 3-class toy matrix returns the expected ordered pairs.

## Acceptance criteria this PRD unblocks

"Threshold-plus-fallback analysis with the resulting accuracy/cost curve"; requirement 4
(confused pairs, does the fallback recover the gap); the resume line's X and Y.

## Open questions

1. Confidence signal: calibrated max-prob (recommended default) vs margin (top1 − top2) vs
   entropy. Cheap to compute all three and plot; pick the best on val.
2. Per-intent view: small intents (80-120 test messages) get noisy numbers (about ±4-5 points).
   Show counts beside every per-intent figure.
