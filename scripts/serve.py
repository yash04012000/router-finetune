"""Run the router playground: one web page + a small JSON API for every model.

    python -m scripts.serve                # then open http://127.0.0.1:8000
    python -m scripts.serve --port 8080
    python -m scripts.serve --verbose      # also print DEBUG lines (the messages typed, top-3 per model)

API (all JSON, local only):
    GET  /api/models              every model, and whether it is available yet
    GET  /api/intents             the 16 intents with their one-line descriptions
    GET  /api/examples?n=6        random test messages with their true intent
    POST /api/predict             {"text": "...", "models": ["tfidf_lr", ...]}  -> one result per model
    GET  /api/scoreboard          accuracy / macro F1 / latency of every approach on the test split
    GET  /api/debug               versions, model files, split hashes, recent errors
    GET  /api/logs?n=200          the last n lines of logs/router.log

Debugging: every request gets an id like [00007]. It is in the console, in logs/router.log, in the
`X-Request-Id` response header and in the Debug tab of the page, so you can follow one request
from the browser to the log. Uses only Python's standard library.
"""

import argparse
import collections
import itertools
import json
import logging
import platform
import random
import sys
import time
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
import sklearn

from router import log
from router.intents import REPO_ROOT, load_intents
from router.scoreboard import score_all
from router.serving import Registry
from router.splits import LOCK_FILE_NAME, file_sha256, load_split

logger = logging.getLogger("router.serve")

PAGE = REPO_ROOT / "ui" / "playground.html"
MAX_TEXT_CHARS = 1000  # a customer message, not a document
MAX_BODY_BYTES = 10_000

registry = Registry()
_test_examples = None  # loaded on first use
_request_ids = itertools.count(1)
_started = time.time()
_requests_served = 0
_recent_errors = collections.deque(maxlen=20)  # shown in the Debug tab


class ClientError(Exception):
    """The caller sent something we can't use (bad JSON, empty text, ...). Answered with a 4xx."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def test_examples():
    global _test_examples
    if _test_examples is None:
        _test_examples = load_split("test")
    return _test_examples


def api_examples(n: int) -> list[dict]:
    chosen = random.sample(test_examples(), min(n, 20))
    return [{"text": e.text, "intent": e.intent} for e in chosen]


def api_predict(body: dict, request_id: str) -> dict:
    text = str(body.get("text", "")).strip()
    if not text:
        raise ClientError("Type a message first.")
    if len(text) > MAX_TEXT_CHARS:
        raise ClientError(f"Message is too long (max {MAX_TEXT_CHARS} characters).")
    names = body.get("models", [])
    logger.debug("[%s] predict text=%r models=%s", request_id, text, names)
    results, errors = [], {}
    for name in names:
        try:
            results.append(registry.predict(name, text).to_dict())
        except KeyError as err:
            errors[name] = str(err.args[0])
    summary = ", ".join(
        f"{r['model']}={r['intent']}({r['confidence']:.2f}, {r['latency_ms']:.1f}ms)" for r in results
    )
    logger.info(
        "[%s] predict %d chars -> %s%s",
        request_id,
        len(text),
        summary or "no results",
        f"  ERRORS {errors}" if errors else "",
    )
    return {"results": results, "errors": errors}


def api_debug() -> dict:
    lock_path = REPO_ROOT / "data" / LOCK_FILE_NAME
    lock = json.loads(lock_path.read_text(encoding="utf-8")) if lock_path.exists() else {}
    splits = {}
    for name, expected in lock.items():
        path = REPO_ROOT / "data" / f"{name}.jsonl"
        actual = file_sha256(path) if path.exists() else None
        splits[name] = {"exists": path.exists(), "hash_ok": actual == expected, "sha256": (actual or "")[:12]}
    return {
        "time": datetime.now(UTC).isoformat(timespec="seconds"),
        "uptime_seconds": round(time.time() - _started),
        "requests_served": _requests_served,
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scikit-learn": sklearn.__version__,
            "platform": platform.platform(),
        },
        "models": registry.debug_info(),
        "splits": splits,
        "log_file": str(log.LOG_FILE),
        "recent_errors": list(_recent_errors),
    }


class Handler(BaseHTTPRequestHandler):
    request_id = "-----"
    status = 0

    # ---- plumbing: one wrapper times every request, logs it, and turns crashes into clean 500s ----
    def handle_request(self, method: str, route) -> None:
        global _requests_served
        self.request_id = f"{next(_request_ids):05d}"
        self.status = 500
        start = time.perf_counter()
        try:
            route()
        except ClientError as err:
            logger.warning("[%s] %s %s -> %d %s", self.request_id, method, self.path, err.status, err)
            self.send_json({"error": str(err)}, err.status)
        except (BrokenPipeError, ConnectionResetError):
            logger.warning("[%s] browser closed the connection before the reply was sent", self.request_id)
        except Exception as err:
            logger.exception("[%s] unhandled error in %s %s", self.request_id, method, self.path)
            _recent_errors.append(
                {"request_id": self.request_id, "when": datetime.now(UTC).isoformat(timespec="seconds"),
                 "request": f"{method} {self.path}", "error": f"{type(err).__name__}: {err}"}
            )  # fmt: skip
            self.send_json(
                {"error": f"Server error ({type(err).__name__}). See the log for request {self.request_id}."},
                500,
            )
        finally:
            _requests_served += 1
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.info(
                "[%s] %s %s -> %d in %.1f ms", self.request_id, method, self.path, self.status, elapsed_ms
            )

    def send_json(self, payload, status: int = 200) -> None:
        if isinstance(payload, dict):
            payload = {**payload, "request_id": self.request_id}
        data = json.dumps(payload).encode("utf-8")
        self.send_bytes(data, "application/json; charset=utf-8", status)

    def send_bytes(self, data: bytes, content_type: str, status: int = 200) -> None:
        self.status = status
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Request-Id", self.request_id)
        self.end_headers()
        self.wfile.write(data)

    # ---- routes ----
    def do_GET(self) -> None:
        self.handle_request("GET", self.route_get)

    def do_POST(self) -> None:
        self.handle_request("POST", self.route_post)

    def route_get(self) -> None:
        url = urlparse(self.path)
        query = parse_qs(url.query)
        if url.path in ("/", "/index.html"):
            self.send_bytes(PAGE.read_bytes(), "text/html; charset=utf-8")
        elif url.path == "/favicon.ico":
            self.send_bytes(b"", "image/x-icon", 204)  # no icon; answer quietly instead of logging a 404
        elif url.path == "/api/models":
            self.send_json(registry.list_models())
        elif url.path == "/api/intents":
            self.send_json([{"name": i.name, "description": i.description} for i in load_intents()])
        elif url.path == "/api/examples":
            self.send_json(api_examples(self.int_param(query, "n", 6)))
        elif url.path == "/api/scoreboard":
            self.send_json(score_all("test"))
        elif url.path == "/api/debug":
            self.send_json(api_debug())
        elif url.path == "/api/logs":
            self.send_json({"path": str(log.LOG_FILE), "lines": log.tail(self.int_param(query, "n", 200))})
        else:
            raise ClientError(f"No such page: {url.path}", 404)

    def route_post(self) -> None:
        if urlparse(self.path).path != "/api/predict":
            raise ClientError(f"No such endpoint: {self.path}", 404)
        length = int(self.headers.get("Content-Length", 0))
        if length > MAX_BODY_BYTES:
            raise ClientError("Request too large.", 413)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError as err:
            raise ClientError(f"Request body is not valid JSON: {err}") from err
        started = time.perf_counter()
        out = api_predict(body, self.request_id)
        out["server_ms"] = round((time.perf_counter() - started) * 1000, 2)
        self.send_json(out)

    @staticmethod
    def int_param(query: dict, name: str, default: int) -> int:
        try:
            return max(1, min(int(query.get(name, [default])[0]), 1000))
        except ValueError as err:
            raise ClientError(f"'{name}' must be a whole number") from err

    def log_message(self, format, *args) -> None:
        pass  # http.server's own access line is replaced by our [request id] line above


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--verbose", action="store_true", help="show DEBUG lines on the console too")
    args = parser.parse_args()
    log.setup_logging(verbose=args.verbose)

    logger.info(
        "python %s, scikit-learn %s, numpy %s", sys.version.split()[0], sklearn.__version__, np.__version__
    )
    for m in registry.debug_info():
        logger.info(
            "model %-13s %s", m["name"], "available" if m["available"] else f"unavailable: {m['reason']}"
        )
    for name, s in api_debug()["splits"].items():
        level = logging.INFO if s["hash_ok"] else logging.ERROR
        logger.log(level, "split %-5s exists=%s hash_ok=%s", name, s["exists"], s["hash_ok"])
    logger.info("log file: %s", log.LOG_FILE)

    # 127.0.0.1 = only this computer can connect.
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    logger.info("Router playground running at http://127.0.0.1:%d  (Ctrl+C to stop)", args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("stopped after %d requests", _requests_served)


if __name__ == "__main__":
    main()
