"""Tests for the playground: the model registry, the scoreboard, and the HTTP API."""

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest
from scripts import serve

from router.scoreboard import score_all
from router.serving import Registry

needs_model = pytest.mark.skipif(
    not (Registry().infos["tfidf_lr"].available), reason="models/tfidf_lr.joblib not trained yet"
)


@pytest.fixture(scope="module")
def base_url():
    server = ThreadingHTTPServer(("127.0.0.1", 0), serve.Handler)  # port 0 = pick any free port
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def get(url):
    with urllib.request.urlopen(url) as r:
        return json.load(r)


def post(url, payload):
    req = urllib.request.Request(url, json.dumps(payload).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.load(r)


def test_registry_lists_planned_models_as_unavailable():
    models = {m["name"]: m for m in Registry().list_models()}
    for planned in ("distilbert", "lora_qwen", "llm_zeroshot", "hybrid"):
        assert models[planned]["available"] is False
        assert models[planned]["planned_in"]


def test_unavailable_model_cannot_be_run():
    with pytest.raises(KeyError):
        Registry().predict("distilbert", "hello")


@needs_model
def test_tfidf_result_is_well_formed():
    r = Registry().predict("tfidf_lr", "I lost my card")
    assert r.intent == r.top[0]["intent"]
    assert 0 <= r.confidence <= 1
    assert [t["prob"] for t in r.top] == sorted((t["prob"] for t in r.top), reverse=True)
    assert len(r.top) == 5


@needs_model
def test_scoreboard_matches_known_tfidf_scores():
    row = next(r for r in score_all("test") if r["approach"] == "tfidf_lr")
    assert row["n"] == 3195
    assert 0.9 < row["accuracy"] <= 1
    assert row["invalid"] == 0


def test_api_models_and_intents(base_url):
    assert {m["name"] for m in get(base_url + "/api/models")} >= {"tfidf_lr", "hybrid"}
    assert len(get(base_url + "/api/intents")) == 16


def test_api_examples_come_with_true_intents(base_url):
    examples = get(base_url + "/api/examples?n=4")
    assert len(examples) == 4 and all(e["text"] and e["intent"] for e in examples)


@needs_model
def test_api_predict(base_url):
    out = post(
        base_url + "/api/predict", {"text": "my card has not arrived", "models": ["tfidf_lr", "distilbert"]}
    )
    assert [r["model"] for r in out["results"]] == ["tfidf_lr"]
    assert "distilbert" in out["errors"]  # asked for a model that is not built yet


@pytest.mark.parametrize(
    "payload", [{"text": "  ", "models": ["tfidf_lr"]}, {"text": "x" * 1001, "models": []}]
)
def test_api_rejects_bad_input(base_url, payload):
    with pytest.raises(urllib.error.HTTPError) as err:
        post(base_url + "/api/predict", payload)
    assert err.value.code == 400


def test_page_is_served(base_url):
    with urllib.request.urlopen(base_url + "/") as r:
        assert b"Router playground" in r.read()
