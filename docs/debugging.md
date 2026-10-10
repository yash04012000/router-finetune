# Debugging guide

Where to look when something looks wrong.

## The log file

Every script and the server write to **`logs/router.log`** (git-ignored, rotated at 1 MB, 3 old files kept).
The file always has DEBUG detail. The console shows INFO and above; add `--verbose` to see DEBUG on the
console too.

```
2026-10-09 18:15:03,002 INFO    router.serve: [00001] predict 32 chars -> tfidf_lr=card_payment_problem(0.98, 13.2ms)
^ time                  ^ level ^ module      ^ request id ^ what happened
```

Module names tell you where a line came from: `router.serve` (HTTP server), `router.serving` (model
registry and loading), `router.tfidf` (training/prediction), `router.splits` (frozen data check),
`router.results` (files written), `router.train_tfidf` / `router.predict` (the scripts).

| Level | Meaning |
|---|---|
| DEBUG | Detail: the message text typed, each model's top-3, which split was verified |
| INFO | Normal progress: requests, model loads, training per C, files written |
| WARNING | Something was refused or odd: bad JSON (400), unknown page (404), asking for a model that is not built yet |
| ERROR / traceback | A real bug: the server logs the full Python traceback and replies 500 |

## Following one request

Every request to the server gets an id like `00007`. It is:

- in the log and console, in square brackets,
- in the `X-Request-Id` response header,
- in every JSON reply (`request_id`) and in error messages shown on the page,
- in the **Debug** tab's call table and the raw request/response panel under the results.

If the page shows "Server error (ValueError). See the log for request 00012", search the log for `[00012]`.

## The Debug tab (http://127.0.0.1:8000, third tab)

- **Health**: versions, uptime, requests served.
- **Models**: available or not, **why not** (for example "model file missing: ...models/tfidf_lr.joblib (train it first)"), and whether it is loaded in memory yet.
- **Data splits**: each file's SHA-256 against `data/splits.lock.json`. `CHANGED` means a split was edited and scores are no longer comparable.
- **Recent server errors**: the last 20 crashes with request id.
- **Server log**: live tail of `logs/router.log` (auto-refresh option); warnings and errors are coloured.
- **This page's API calls**: the last 30 calls from your browser tab with status and time. The same lines go to the browser console (F12, level "Verbose").

Raw endpoints for the same data: `GET /api/debug` and `GET /api/logs?n=200`.

## Common problems

| Symptom | Likely cause | Check |
|---|---|---|
| A model is greyed out | Model file missing or its PRD is not built | Debug tab, Models table, "Why not" |
| `ValueError: ... has changed since it was frozen` | A `data/*.jsonl` file was edited | Debug tab, Data splits; `git diff data/` |
| First request after start is slow (about 0.4 s) | The model is loaded on first use | Log line "first use of ...: loading it now" |
| First prediction latency looks high (10+ ms vs 0.7 ms) | Warm-up of the first call | Compare with the second call; PRD 4 measures latency after 20 warm-up calls |
| Lab tab says "Lab unavailable: PyTorch and transformers are not installed" (HTTP 503) | Training libraries missing, or the model not downloaded and no internet | `pip install -r requirements-train.txt`; run `python -m scripts.explore_model` once to download |
| Lab: first click takes about 6 s | The tokenizer and model load on first use | Log lines `lab: tokenizer loaded`, `lab: untrained model loaded` |
| Page says "Could not reach the server" | Server stopped or another port | Terminal where `scripts.serve` runs |
| Port already in use | An old server is still running | `python -m scripts.serve --port 8001` |

## Scripts

```
python -m scripts.train_tfidf --verbose
python -m scripts.predict --approach tfidf_lr --splits val --verbose
python -m scripts.serve --verbose
```

All three log to the console and `logs/router.log`. New code should use
`logger = logging.getLogger("router.<name>")` and `logger.info/debug/warning` (no `print`), and call
`log.setup_logging()` once at the top of a new script.
