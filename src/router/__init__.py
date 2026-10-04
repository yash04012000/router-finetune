"""router-finetune: learn fine-tuning by building a support-intent router.

Modules, in the order you'd read them:
    intents  - the list of possible answers (config/intents.yaml)
    schema   - what one example and one prediction look like
    jsonl    - read/write one-JSON-object-per-line files
    splits   - load the frozen train/val/test files (and refuse if they changed)
    metrics  - accuracy, F1, confusion matrix, latency percentiles, confidence intervals
    cost     - cost per 1,000 routing decisions
"""
