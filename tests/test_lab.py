"""Tests for the Lab tab backend (src/router/lab.py) and its API endpoints.

The model views need torch, transformers and the downloaded DistilBERT files; they are skipped otherwise
(run `python -m scripts.explore_model` once to download).
"""

import json
import math
import urllib.error
import urllib.request

import pytest

from router import lab


def post(url, payload):
    req = urllib.request.Request(url, json.dumps(payload).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.load(r)


def get(url):
    with urllib.request.urlopen(url) as r:
        return json.load(r)


def status_of(url, payload=None):
    """HTTP status and JSON body for a request, without raising on 4xx/5xx."""
    try:
        return 200, (post(url, payload) if payload is not None else get(url))
    except urllib.error.HTTPError as err:
        return err.code, json.load(err)


def lab_ready() -> bool:
    if not lab.libraries_ok():
        return False
    try:
        lab.run_model("hello")
        return True
    except lab.LabUnavailable:
        return False


needs_lab = pytest.mark.skipif(not lab_ready(), reason="torch/transformers/DistilBERT files not available")


# ---------- always run: behaviour when the Lab cannot run ----------
def test_lab_reports_unavailable_when_libraries_are_missing(monkeypatch):
    monkeypatch.setattr(lab, "libraries_ok", lambda: False)
    monkeypatch.setattr(lab, "_loaded", {})
    with pytest.raises(lab.LabUnavailable, match="requirements-train.txt"):
        lab.tokenize("hello")


def test_api_answers_503_not_500_when_lab_is_unavailable(base_url, monkeypatch):
    monkeypatch.setattr(lab, "libraries_ok", lambda: False)
    monkeypatch.setattr(lab, "_loaded", {})
    status, body = status_of(base_url + "/api/lab/tokenize", {"text": "hello"})
    assert status == 503 and "not installed" in body["error"]


def test_lab_endpoints_validate_the_text(base_url):
    status, body = status_of(base_url + "/api/lab/model", {"text": "   "})
    assert status == 400 and "Type a message" in body["error"]


# ---------- need the model ----------
@needs_lab
def test_tokenize_marks_special_tokens_and_continuations():
    out = lab.tokenize("I recieved the wrong amout")
    tokens = [p["token"] for p in out["pieces"]]
    assert tokens[0] == "[CLS]" and tokens[-1] == "[SEP]"
    assert out["pieces"][0]["special"] and out["pieces"][0]["id"] == 101
    assert any(p["continues"] for p in out["pieces"])  # '##ie' etc.
    assert out["n_tokens"] == len(out["pieces"]) and out["n_words"] == 5


@needs_lab
def test_tokenize_reports_what_would_be_truncated():
    short = lab.tokenize("my card")
    assert short["truncated_tokens"] == 0
    long = lab.tokenize("word " * 100)
    assert long["truncated_tokens"] == long["n_tokens"] - long["max_len"] > 0


@needs_lab
def test_run_model_shapes_probabilities_and_the_by_hand_check():
    out = lab.run_model("My card was declined at the shop")
    n = out["shapes"]["input_ids"][1]
    assert out["shapes"] == {
        "input_ids": [1, n],
        "hidden_states": [1, n, 768],
        "cls_vector": [1, 768],
        "logits": [1, 16],
    }
    assert len(out["classes"]) == 16 and len(out["cls_preview"]) == lab.CLS_PREVIEW
    assert out["prob_sum"] == pytest.approx(1.0, abs=1e-4)
    assert out["head_matches_by_hand"] is True
    assert out["uniform_loss"] == pytest.approx(math.log(16))
    assert out["trained"] is False


@needs_lab
def test_untrained_model_is_nearly_flat_on_any_message():
    out = lab.run_model("I would like to change my PIN")
    assert max(c["prob"] for c in out["classes"]) < 0.15  # a guess: 1/16 = 0.0625


@needs_lab
def test_same_text_gives_the_same_answer_every_time():
    assert lab.run_model("hello there")["classes"] == lab.run_model("hello there")["classes"]


@needs_lab
def test_model_info_accounts_for_every_parameter():
    info = lab.model_info()
    assert info["total"] == 66_965_776 == info["body"] + info["head"]
    assert info["head"] == 602_896
    assert sum(p["params"] for p in info["parts"]) == info["total"]


@needs_lab
def test_lab_api_endpoints(base_url):
    tok = post(base_url + "/api/lab/tokenize", {"text": "my card was declined"})
    assert tok["pieces"][0]["token"] == "[CLS]" and "request_id" in tok
    model = post(base_url + "/api/lab/model", {"text": "my card was declined"})
    assert len(model["classes"]) == 16
    info = get(base_url + "/api/lab/info")
    assert info["total"] == 66_965_776
