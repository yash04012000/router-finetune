# PRD 5 — Large model, prompted zero-shot

Status: Not started · Depends on: PRDs 1-2 (build after PRD 4) · Blocks: PRDs 6-7

## Summary

The "what most teams do today" baseline: a large hosted model, given the intent list and
descriptions, asked to route each conversation. It's both a comparison row and the fallback
target in the hybrid, so it has to run on val *and* test, with real token counts captured for
costing. Responses are cached so the numbers reproduce without an API key.

## Goals

- One prompt template, versioned in `config/prompts/zeroshot_v1.txt`, containing the 16 intents
  with descriptions; no examples (zero-shot, per the spec).
- Structured output: the model returns exactly one intent label (JSON `{"intent": "..."}` or
  provider-native structured output / tool call when available).
- Run on val and test; capture input/output tokens and wall latency per call.
- Disk cache keyed on (model, prompt version, example id, params) → committed, so re-runs and CI
  never call the API.
- Optional second model (a cheaper "mid-tier" hosted model) as an extra row, if budget allows.

## Non-goals

- Few-shot or chain-of-thought prompting variants (would muddy "zero-shot"; mention in DESIGN.md
  as a cheap thing a team could try first).
- Building a production router service.

## Design

### Client — `src/router/llm/client.py`

LiteLLM `completion()` wrapped in a narrow function (same pattern as agent-eval-harness):
`classify(example) -> (label, raw_text, in_tok, out_tok, latency_ms)`. temperature 0, max output
tokens ~20, one retry on transient errors with backoff. Calls run with modest concurrency
(e.g. 8) for throughput, but **latency is the per-call wall time**, recorded per request.

Latency caveat recorded in the meta file and README: API latency includes network and provider
queueing, measured from one location at one time of day. Run the test set twice on different
days and report both p95s if they differ materially.

### Parsing

- Accept the label if it exactly matches an intent (case/whitespace-normalized).
- Otherwise map to `__invalid__` (counts as wrong). Count of invalid outputs reported as its own
  number — it's part of the honest picture of prompting vs fine-tuning.

### Confidence

Not needed for the hybrid (the large model is the fallback, never the thing being thresholded).
If the provider exposes logprobs, record them in `probs` for interest; otherwise `confidence=None`.

### Cost

From recorded tokens and `config/pricing.yaml` (PRD 1). The prompt (intent list) is the same on
every call, so note in DESIGN.md that prompt caching would cut input cost substantially; report
the uncached number as the headline and the prompt-cached estimate as a secondary line, clearly
labelled.

```
config/prompts/zeroshot_v1.txt
src/router/llm/client.py, parse.py, cache.py
scripts/run_llm.py --model <id> --splits val,test
results/llm_cache/<model>/<hash>.json
tests/test_llm_parse.py, test_llm_cache.py
```

## Testing plan

- Parsing: exact label, label with trailing punctuation, JSON-wrapped label, a label for a
  non-existent intent → `__invalid__`, empty output → `__invalid__`.
- Cache: second run over the same examples makes zero network calls (stub transport counter);
  changing the prompt version invalidates the key.
- Prompt render: every intent in `intents.yaml` appears exactly once.

## Acceptance criteria this PRD unblocks

"Large model, prompted with the intent list, zero-shot" leg; the fallback predictions PRD 6 needs.

## Open questions

1. Which large model. Recommend a current frontier-tier hosted model for the main row (the
   "expensive default" the story argues against), and if budget allows a mid-tier model as a
   second row — that comparison is often more interesting to cost-sensitive teams. Must be a
   different family from any model used to build the data (none now, data is public).
2. Budget: ~4,250 calls (val 1,053 + test 3,195) × ~600 input tokens. **No paid API budget is
   available**, so decide before starting this PRD: a free-tier hosted model, or a local model
   through Ollama (an 8B-class instruct model fits an 8 GB GPU quantized). Cost per 1k is then
   computed from tokens at the model's published list price, and clearly labelled as such.
   To keep spend at zero the test-set run can use a fixed random subsample (e.g. 1,000) if rate limits bite.
