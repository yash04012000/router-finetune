# P5: Small-model router fine-tune

Repo: `router-finetune` | Effort: 2 weeks | Unlocks: fine-tuning roles, cost-sensitive GenAI teams

## The problem

Routing a conversation to the right specialist agent is a classification problem, and most teams pay
a large model to do it on every turn. Show that a small fine-tuned model does the same job at a
fraction of the cost and latency, and quantify exactly where it stops being good enough.

## Functional requirements

1. Dataset: 5,000+ examples across 10-20 support intents, published with the repo. Synthetic is fine
   if stated, with the generation script included.
2. Three approaches compared on the same test set:
   - Large model, prompted with the intent list, zero-shot
   - Small open model, LoRA / QLoRA fine-tuned
   - A classical baseline (fine-tuned DistilBERT, or TF-IDF + logistic regression)
3. Report per approach: accuracy and macro F1, p95 latency, cost per 1,000 routing decisions,
   confusion matrix.
4. Analyse where the small model fails: which intent pairs it confuses, and whether a confidence
   threshold with fallback to the large model recovers the gap. This hybrid result is the interesting
   finding: lead with it.
5. Training reproducibility: script, exact hyperparameters, seed, hardware.

## Acceptance criteria

- [ ] Three approaches evaluated on an identical held-out test set
- [ ] Cost and latency reported next to accuracy for each
- [ ] Threshold-plus-fallback analysis with the resulting accuracy/cost curve
- [ ] Adapter weights or a link to them, plus a one-command inference example
- [ ] Honest limitations section covering dataset realism

## Resume line it earns

Fine-tuned a small open model (LoRA) for support-intent routing and benchmarked it against a prompted
large model and a Transformer baseline, matching accuracy within X% at Y% of the cost with
confidence-based fallback.
