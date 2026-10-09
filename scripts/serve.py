"""Run the router playground: one web page + a small JSON API for every model.

    python -m scripts.serve                # then open http://127.0.0.1:8000
    python -m scripts.serve --port 8080

API (all JSON, local only):
    GET  /api/models              every model, and whether it is available yet
    GET  /api/intents             the 16 intents with their one-line descriptions
    GET  /api/examples?n=6        random test messages with their true intent
    POST /api/predict             {"text": "...", "models": ["tfidf_lr", ...]}  -> one result per model
    GET  /api/scoreboard          accuracy / macro F1 / latency of every approach on the test split

Uses only Python's standard library, so there is nothing extra to install.
"""

import argparse
import json
import random
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from router.intents import REPO_ROOT, load_intents
from router.scoreboard import score_all
from router.serving import Registry
from router.splits import load_split

PAGE = REPO_ROOT / "ui" / "playground.html"
MAX_TEXT_CHARS = 1000  # a customer message, not a document
MAX_BODY_BYTES = 10_000

registry = Registry()
_test_examples = None  # loaded on first use


def test_examples():
    global _test_examples
    if _test_examples is None:
        _test_examples = load_split("test")
    return _test_examples


def api_examples(n: int) -> list[dict]:
    chosen = random.sample(test_examples(), min(n, 20))
    return [{"text": e.text, "intent": e.intent} for e in chosen]


def api_predict(body: dict) -> dict:
    text = str(body.get("text", "")).strip()
    if not text:
        raise ValueError("Type a message first.")
    if len(text) > MAX_TEXT_CHARS:
        raise ValueError(f"Message is too long (max {MAX_TEXT_CHARS} characters).")
    results, errors = [], {}
    for name in body.get("models", []):
        try:
            results.append(registry.predict(name, text).to_dict())
        except KeyError as err:
            errors[name] = str(err.args[0])
    return {"results": results, "errors": errors}


class Handler(BaseHTTPRequestHandler):
    def send_json(self, payload, status: int = 200) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        url = urlparse(self.path)
        if url.path in ("/", "/index.html"):
            data = PAGE.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif url.path == "/api/models":
            self.send_json(registry.list_models())
        elif url.path == "/api/intents":
            self.send_json([{"name": i.name, "description": i.description} for i in load_intents()])
        elif url.path == "/api/examples":
            n = int(parse_qs(url.query).get("n", ["6"])[0])
            self.send_json(api_examples(n))
        elif url.path == "/api/scoreboard":
            self.send_json(score_all("test"))
        else:
            self.send_json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/predict":
            return self.send_json({"error": "not found"}, 404)
        length = int(self.headers.get("Content-Length", 0))
        if length > MAX_BODY_BYTES:
            return self.send_json({"error": "request too large"}, 413)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
            self.send_json(api_predict(body))
        except (ValueError, json.JSONDecodeError) as err:
            self.send_json({"error": str(err)}, 400)

    def log_message(self, format, *args) -> None:
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    # 127.0.0.1 = only this computer can connect.
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Router playground running at http://127.0.0.1:{args.port}  (Ctrl+C to stop)")
    server.serve_forever()


if __name__ == "__main__":
    main()
