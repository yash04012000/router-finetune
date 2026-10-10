"""The "Lab": live views of what happens inside the fine-tuning pipeline, for the playground's Lab tab.

Each DistilBERT step adds a view here, so you can poke at the idea instead of only reading about it:
    step 3.2  tokenize()      your message as tokens, ids and attention mask
    step 3.3  model_info()    where the 66,965,776 parameters are
              run_model()     one message through the UNTRAINED model: shapes, [CLS] vector, logits, probabilities

torch and transformers are heavy and optional, so they are imported only when a Lab view is used.
If they are missing (for example in CI) the Lab raises LabUnavailable and the server answers with a clear 503.
"""

import importlib.util
import logging
import math
import threading
import time

from router.intents import intent_names

logger = logging.getLogger("router.lab")

CLS_PREVIEW = 96  # how many numbers of the 768-number [CLS] vector the UI draws

_lock = threading.Lock()
_loaded: dict = {}


class LabUnavailable(Exception):
    """The Lab cannot run here (libraries missing, or the model files are not downloaded)."""


def libraries_ok() -> bool:
    return all(importlib.util.find_spec(name) is not None for name in ("torch", "transformers"))


def _tokenizer():
    if not libraries_ok():
        raise LabUnavailable(
            "PyTorch and transformers are not installed. Run: pip install -r requirements-train.txt"
        )
    with _lock:
        if "tokenizer" not in _loaded:
            from router.baselines import distilbert as db

            start = time.perf_counter()
            try:
                _loaded["tokenizer"] = db.load_tokenizer()
            except OSError as err:
                raise LabUnavailable(
                    f"Could not load the tokenizer (offline and not cached?): {err}"
                ) from err
            logger.info("lab: tokenizer loaded in %.0f ms", (time.perf_counter() - start) * 1000)
    return _loaded["tokenizer"]


def _model():
    """The untrained model: pretrained body + a random head (seed 42, so the same every time)."""
    _tokenizer()  # also checks the libraries
    with _lock:
        if "model" not in _loaded:
            import torch

            from router.baselines import distilbert as db

            start = time.perf_counter()
            try:
                model = db.build_model(intent_names())
            except OSError as err:
                raise LabUnavailable(f"Could not load the model (offline and not cached?): {err}") from err
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _loaded["model"] = model.to(
                device
            ).eval()  # eval(): dropout off, so the same text gives the same answer
            _loaded["device"] = device
            logger.info("lab: untrained model loaded on %s in %.1f s", device, time.perf_counter() - start)
    return _loaded["model"], _loaded["device"]


# ---------- step 3.2 ----------
def tokenize(text: str) -> dict:
    """Your message as the model receives it."""
    from router.baselines import distilbert as db

    tokenizer = _tokenizer()
    ids = tokenizer(text)["input_ids"]  # no truncation here, so we can show what WOULD be cut
    tokens = tokenizer.convert_ids_to_tokens(ids)
    special = set(tokenizer.all_special_tokens)
    pieces = [
        {"token": t, "id": i, "special": t in special, "continues": t.startswith("##")}
        for t, i in zip(tokens, ids, strict=True)
    ]
    n_tokens = len(ids)
    logger.debug("lab tokenize: %d words -> %d tokens", len(text.split()), n_tokens)
    return {
        "pieces": pieces,
        "n_words": len(text.split()),
        "n_tokens": n_tokens,
        "max_len": db.MAX_LEN,
        "truncated_tokens": max(0, n_tokens - db.MAX_LEN),
        "vocab_size": tokenizer.vocab_size,
    }


# ---------- step 3.3 ----------
def model_info() -> dict:
    """Where the parameters are: body (embeddings, 6 layers) and the new head."""
    from router.baselines import distilbert as db

    model, device = _model()
    layer = db.count_parameters(model.distilbert.transformer.layer[0])
    parts = [
        {"name": "Embeddings", "detail": "30,522 words x 768 + 512 positions x 768", "params": db.count_parameters(model.distilbert.embeddings), "group": "body"},
        {"name": "6 transformer layers", "detail": f"6 x {layer:,}", "params": 6 * layer, "group": "body"},
        {"name": "pre_classifier", "detail": "Linear 768 -> 768", "params": db.count_parameters(model.pre_classifier), "group": "head"},
        {"name": "classifier", "detail": "Linear 768 -> 16", "params": db.count_parameters(model.classifier), "group": "head"},
    ]  # fmt: skip
    total = db.count_parameters(model)
    assert sum(p["params"] for p in parts) == total  # every parameter is accounted for
    return {
        "total": total,
        "body": db.count_parameters(model.distilbert),
        "head": db.head_parameters(model),
        "parts": parts,
        "device": device,
        "weights_mb": round(total * 4 / 1e6),
    }


def run_model(text: str) -> dict:
    """One message through the untrained model, with every shape on the way."""
    import torch
    import torch.nn.functional as F

    from router.baselines import distilbert as db

    tokenizer = _tokenizer()
    model, device = _model()
    labels = intent_names()
    batch = tokenizer(text, truncation=True, max_length=db.MAX_LEN, return_tensors="pt")
    batch = {k: v.to(device) for k, v in batch.items()}

    start = time.perf_counter()
    with torch.no_grad():
        out = model(**batch, output_hidden_states=True)
        hidden = out.hidden_states[-1]
        cls_vec = hidden[:, 0]
        by_hand = model.classifier(torch.relu(model.pre_classifier(cls_vec)))  # the head, rebuilt
    if device == "cuda":
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - start) * 1000

    logits = out.logits[0].cpu()
    probs = F.softmax(logits, dim=0)
    order = torch.argsort(probs, descending=True)
    n_tokens = batch["input_ids"].shape[1]
    logger.debug(
        "lab run_model: %d tokens, top=%s (%.3f), %.1f ms",
        n_tokens,
        labels[order[0]],
        probs[order[0]],
        latency_ms,
    )
    return {
        "device": device,
        "latency_ms": round(latency_ms, 2),
        "shapes": {
            "input_ids": [1, n_tokens],
            "hidden_states": [1, n_tokens, 768],
            "cls_vector": [1, 768],
            "logits": [1, len(labels)],
        },
        "cls_preview": [round(x, 3) for x in cls_vec[0, :CLS_PREVIEW].cpu().tolist()],
        "classes": [
            {"intent": labels[i], "logit": round(logits[i].item(), 4), "prob": round(probs[i].item(), 5)}
            for i in range(len(labels))
        ],
        "top": [{"intent": labels[i], "prob": round(probs[i].item(), 5)} for i in order[:3].tolist()],
        "prob_sum": round(probs.sum().item(), 6),
        "uniform_prob": 1 / len(labels),
        "uniform_loss": math.log(len(labels)),
        "head_matches_by_hand": bool(torch.allclose(by_hand, out.logits, atol=1e-5)),
        "trained": False,
    }
