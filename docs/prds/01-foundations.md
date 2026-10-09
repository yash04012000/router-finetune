# PRD 1 — Foundations: taxonomy, schemas, predictions contract, metrics, cost model

Status: Not started · Depends on: nothing · Blocks: PRDs 2-7

## Summary

Three approaches will be compared, then combined into a hybrid. That comparison is only honest
if every approach is scored by the same code from the same kind of output. This PRD defines the
intent taxonomy, the per-example predictions file every approach must write, and the metric and
cost functions that turn those files into numbers. Approach-specific code ends at "write a
predictions file"; everything downstream is shared.

## Goals

- Fix the intent taxonomy (16 intents, with deliberately confusable pairs) in one config file.
- Define `Example` and `Prediction` Pydantic models and the JSONL predictions format.
- Implement metrics: accuracy, macro F1, per-class P/R/F1, confusion matrix, p50/p95 latency,
  bootstrap 95% CIs for accuracy and macro F1.
- Implement the cost model: cost per 1,000 routing decisions for API (token-priced) and
  self-hosted (GPU/CPU-hour-priced) approaches.
- Pin the environment, set up the package layout, and get CI (ruff + pytest) green.

## Non-goals

- Producing any data (PRD 2) or predictions (PRDs 3-5).
- Plots and README tables (PRD 7).

## Design

### 1. Intent taxonomy — `config/intents.yaml`

16 intents for a generic e-commerce/subscription support desk, each with a one-line description
(used verbatim in the large-model prompt in PRD 5 and in the generation prompt in PRD 2):

| Intent | Designed to be confused with |
|---|---|
| `billing_dispute` | `fraud_security`, `refund_request` |
| `refund_request` | `return_exchange`, `billing_dispute` |
| `payment_method_update` | `account_update` |
| `subscription_cancel` | `subscription_change` |
| `subscription_change` | `subscription_cancel` |
| `order_status` | `damaged_or_wrong_item` |
| `return_exchange` | `refund_request`, `damaged_or_wrong_item` |
| `damaged_or_wrong_item` | `return_exchange` |
| `account_access` | `fraud_security`, `technical_issue` |
| `account_update` | `account_deletion_privacy`, `payment_method_update` |
| `account_deletion_privacy` | `account_update` |
| `technical_issue` | `account_access` |
| `product_question` | `subscription_change` |
| `complaint_escalation` | most others (angry tone + a concrete issue) |
| `fraud_security` | `account_access`, `billing_dispute` |
| `out_of_scope` | — (chit-chat, unrelated requests) |

The confusable pairs are the point: they give PRD 6 something real to find. They're declared in
the YAML so PRD 2 can oversample them and PRD 6 can check whether the predicted confusions match
the designed ones.

### 2. Schemas — `src/router/schema.py`

```python
class Turn(BaseModel):
    role: Literal["user", "agent"]
    text: str


class Example(BaseModel):
    id: str  # stable, derived from the text, e.g. "banking77-0019cec4b2"
    intent: str  # validated against intents.yaml
    turns: list[Turn]  # last turn is the one being routed; >1 turn = multi-turn context
    source: str  # "banking77" or "clinc150"
    group_id: str = ""  # unused for public data (kept for datasets with paraphrase groups)
    meta: dict = {}  # e.g. original_label (the Banking77 label before merging)


class Prediction(BaseModel):
    example_id: str
    approach: str  # "tfidf_lr" | "distilbert" | "llm_zeroshot" | "lora_<model>"
    predicted: str  # an intent, or "__invalid__" if the model emitted garbage
    confidence: float | None  # max softmax prob (calibrated where noted); None if unavailable
    probs: dict[str, float] | None
    latency_ms: float  # wall time for this single decision, batch size 1
    input_tokens: int | None  # API approaches only
    output_tokens: int | None
```

Predictions files: `results/predictions/<approach>__<split>.jsonl`, plus a sidecar
`<approach>__<split>.meta.json` recording model id/revision, hardware, library versions, git
SHA, date, and the test-set hash it was run against.

`router.data.load_split("test")` verifies the split file's SHA-256 against
`data/splits.lock.json` and refuses to load on mismatch. The metrics loader refuses a
predictions file whose recorded test-set hash differs or whose example ids don't exactly cover
the split. That's how "identical held-out test set" is enforced rather than promised.

### 3. Metrics — `src/router/metrics.py`

- `accuracy`, `macro_f1`, `per_class_report`, `confusion_matrix` via scikit-learn, with labels
  fixed to the taxonomy order so matrices from different approaches line up.
- `__invalid__` counts as wrong for every metric (it's a real failure mode of the LLM).
- `latency_summary(preds) -> {p50, p95, p99, mean}` in ms.
- `bootstrap_ci(metric, y_true, y_pred, n=1000, seed=0)` → (lo, hi). With ~1,300 test examples
  the CI is about ±1-2 points; differences smaller than that get reported as "no clear difference".

### 4. Cost model — `src/router/cost.py`, `config/pricing.yaml`

```yaml
api:
  <model-id>: {input_per_mtok: ..., output_per_mtok: ..., source: <url>, read_on: <date>}
compute:
  gpu_hour_usd: ...      # cloud on-demand price for a comparable GPU, with source + date
  cpu_hour_usd: ...
```

- API: `cost_per_1k = 1000 * mean(in_tok * p_in + out_tok * p_out)` from actual recorded tokens.
- Self-hosted: `cost_per_1k = hour_rate * (1000 / throughput_per_hour)` using measured
  **batched** throughput (the realistic serving cost), reported alongside the batch-1 figure so
  the assumption is visible. Training cost is reported separately as a one-off, not amortized
  into the per-1k number (stated explicitly in the README).

### 5. Environment and layout

- Python 3.11 (matches CI). `requirements.txt` = light deps needed by CI and the bench
  (pydantic, scikit-learn, numpy, pandas, pyyaml, matplotlib, litellm, pytest, ruff), all pinned.
  `requirements-train.txt` = torch, transformers, peft, datasets, accelerate, bitsandbytes,
  pinned. Tests that need torch are marked and skipped when it isn't installed, so CI stays fast.
- `pyproject.toml` for ruff config and making `src/router` importable.

```
config/intents.yaml, config/pricing.yaml
src/router/__init__.py, schema.py, data.py, metrics.py, cost.py, io.py
tests/test_schema.py, test_metrics.py, test_cost.py, test_io.py
requirements.txt, requirements-train.txt, pyproject.toml
```

## Testing plan

- Schema: unknown intent rejected; empty `turns` rejected; round-trip JSONL write/read is lossless.
- Metrics: hand-computed toy cases for accuracy/macro F1; confusion matrix label order is fixed
  even when a class is absent from predictions; `__invalid__` counted wrong; bootstrap CI is
  deterministic for a given seed and contains the point estimate.
- Cost: API and compute formulas against hand-computed values; missing price key raises a clear
  error instead of silently costing $0.
- IO: predictions file with a missing example id, a duplicate id, or a wrong test hash is rejected.

## Acceptance criteria this PRD unblocks

"Cost and latency reported next to accuracy for each" (the functions), "identical held-out test
set" (hash enforcement). Not checkable on its own.

## Open questions

1. 16 intents vs fewer: more intents make the classical baseline's job harder and the confusion
   story richer; recommend keeping 16.
2. Report latency at batch 1 only, or also batched? Recommend: p95 at batch 1 is the headline
   (routing is per-turn, online); batched throughput only feeds the cost model.
