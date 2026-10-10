"""Tests for step 3.3: the DistilBERT body + new head, and the maths on docs/math/fine-tuning.md.

Skipped when torch/transformers are missing or the model has not been downloaded yet
(run `python -m scripts.explore_model` once), so CI and offline runs stay green.
"""

import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

import torch.nn.functional as F

from router.baselines import distilbert as db
from router.format import render
from router.intents import intent_names
from router.splits import load_split

LABELS = intent_names()


@pytest.fixture(scope="module")
def tokenizer():
    try:
        return db.AutoTokenizer.from_pretrained(db.MODEL_NAME, local_files_only=True)
    except OSError:
        pytest.skip("tokenizer not downloaded yet")


@pytest.fixture(scope="module")
def model():
    try:
        m = db.DistilBertForSequenceClassification.from_pretrained(
            db.MODEL_NAME,
            local_files_only=True,
            num_labels=len(LABELS),
            id2label=dict(enumerate(LABELS)),
            label2id={n: i for i, n in enumerate(LABELS)},
        )
    except OSError:
        pytest.skip("model not downloaded yet")
    return m.eval()


@pytest.fixture(scope="module")
def batch(tokenizer):
    texts = [render(e) for e in load_split("val")[:8]]
    return tokenizer(texts, padding=True, truncation=True, max_length=db.MAX_LEN, return_tensors="pt")


# ---------- the maths in docs/math/fine-tuning.md ----------
def test_linear_worked_example():
    x = torch.tensor([1.0, 2.0])
    w = torch.tensor([[0.5, -1.0], [1.0, 0.5], [-0.5, 2.0]])
    b = torch.tensor([0.1, 0.0, -0.2])
    z = w @ x + b
    assert z.tolist() == pytest.approx([-1.4, 2.0, 3.3])
    assert torch.relu(z).tolist() == pytest.approx([0.0, 2.0, 3.3])


def test_softmax_and_cross_entropy_worked_example():
    z = torch.tensor([1.2, -0.3, 0.5])
    p = torch.exp(z) / torch.exp(z).sum()
    assert p.tolist() == pytest.approx([0.5815, 0.1297, 0.2888], abs=1e-4)
    assert p.sum().item() == pytest.approx(1.0)
    assert -math.log(p[2].item()) == pytest.approx(1.2422, abs=1e-4)
    assert -math.log(p[0].item()) == pytest.approx(0.5422, abs=1e-4)
    assert F.cross_entropy(z[None], torch.tensor([2])).item() == pytest.approx(1.2422, abs=1e-4)


def test_softmax_ignores_a_shift_and_keeps_the_winner():
    z = torch.tensor([1.2, -0.3, 0.5])
    assert torch.allclose(F.softmax(z, 0), F.softmax(z + 100.0, 0))
    assert F.softmax(z, 0).argmax() == z.argmax()


def test_uniform_guessing_has_loss_ln_16():
    assert F.cross_entropy(torch.zeros(1, 16), torch.tensor([3])).item() == pytest.approx(math.log(16))


# ---------- the model ----------
def test_parameter_counts_match_the_table_in_the_docs(model):
    assert db.head_parameters(model) == 768 * 768 + 768 + 768 * 16 + 16 == 602_896
    assert db.count_parameters(model.distilbert.embeddings) == 23_835_648
    assert db.count_parameters(model.distilbert.transformer.layer[0]) == 7_087_872
    assert db.count_parameters(model.distilbert) == 66_362_880
    assert db.count_parameters(model) == 66_965_776


def test_label_order_is_the_intents_yaml_order(model):
    assert model.config.num_labels == 16
    assert [model.config.id2label[i] for i in range(16)] == LABELS


def test_shapes_through_the_model(model, batch):
    n, length = batch["input_ids"].shape
    with torch.no_grad():
        out = model(**batch, output_hidden_states=True)
    assert out.hidden_states[-1].shape == (n, length, 768)
    assert out.logits.shape == (n, 16)


def test_the_head_is_pre_classifier_relu_classifier_on_the_cls_vector(model, batch):
    with torch.no_grad():
        out = model(**batch, output_hidden_states=True)
        cls_vector = out.hidden_states[-1][:, 0]
        by_hand = model.classifier(torch.relu(model.pre_classifier(cls_vector)))
    assert torch.allclose(by_hand, out.logits, atol=1e-5)


def test_eval_mode_is_repeatable_and_train_mode_is_not(model, batch):
    try:
        model.eval()
        with torch.no_grad():
            assert torch.equal(model(**batch).logits, model(**batch).logits)
        model.train()
        with torch.no_grad():
            assert not torch.equal(model(**batch).logits, model(**batch).logits)  # dropout
    finally:
        model.eval()


def test_the_same_seed_gives_the_same_new_head():
    a, b = db.build_model(LABELS, seed=7), db.build_model(LABELS, seed=7)
    c = db.build_model(LABELS, seed=8)
    assert torch.equal(a.classifier.weight, b.classifier.weight)
    assert not torch.equal(a.classifier.weight, c.classifier.weight)


def test_untrained_model_is_near_chance_with_loss_near_ln_16(model, tokenizer):
    val = load_split("val")[:64]
    texts, gold = [render(e) for e in val], torch.tensor([LABELS.index(e.intent) for e in val])
    batch = tokenizer(texts, padding=True, truncation=True, max_length=db.MAX_LEN, return_tensors="pt")
    with torch.no_grad():
        logits = model(**batch).logits
    assert 2.5 < F.cross_entropy(logits, gold).item() < 3.1  # ln(16) = 2.77
    assert (logits.argmax(1) == gold).float().mean().item() < 0.3  # nowhere near trained
