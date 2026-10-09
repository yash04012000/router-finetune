"""Tests for PRD 3 (TF-IDF part): render, training, save/load, the predictions it writes."""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from router.baselines import tfidf
from router.format import render
from router.schema import Example, Turn
from router.splits import check_predictions_cover

LABELS = ["cards", "transfers"]


def make(i, intent, text):
    return Example(id=f"e{i}", intent=intent, turns=[Turn("user", text)])


def toy_data():
    cards = ["my card is lost", "card not working", "I need a new card", "card was stolen"]
    transfers = ["send money abroad", "transfer to a friend", "money transfer is late", "wire money today"]
    examples = [make(i, "cards", t) for i, t in enumerate(cards * 3)]
    examples += [make(100 + i, "transfers", t) for i, t in enumerate(transfers * 3)]
    return examples


# ---------- render ----------
def test_render_single_turn_is_just_the_message():
    assert render(make(1, "cards", "hello there")) == "hello there"


def test_render_multi_turn_keeps_context_with_role_tags():
    e = Example(id="x", intent="cards", turns=[Turn("agent", "How can I help?"), Turn("user", "card lost")])
    assert render(e) == "[agent] How can I help? [user] card lost"


# ---------- tf-idf maths (the worked example in docs/math/tfidf-logreg.md) ----------
def test_tfidf_matches_the_worked_example():
    docs = ["card not working", "card payment failed", "top up failed"]
    vec = TfidfVectorizer(sublinear_tf=True)
    x = vec.fit_transform(docs).toarray()[0]
    words = list(vec.get_feature_names_out())
    assert round(vec.idf_[words.index("card")], 4) == 1.2877
    assert round(vec.idf_[words.index("not")], 4) == 1.6931
    assert round(x[words.index("card")], 4) == 0.4736
    assert round(x[words.index("not")], 4) == 0.6228
    assert np.isclose(np.linalg.norm(x), 1.0)  # each vector is length 1


def test_char_chunks_survive_a_typo_but_whole_words_do_not():
    chars = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3)).build_analyzer()
    words = TfidfVectorizer().build_analyzer()
    assert len(set(chars("received")) & set(chars("recieved"))) == 4
    assert set(words("received")) & set(words("recieved")) == set()


# ---------- training + prediction ----------
def test_trains_on_toy_data_and_beats_chance():
    examples = toy_data()
    model, grid = tfidf.train(examples, examples, LABELS)
    assert len(grid) == len(tfidf.C_GRID)
    preds = tfidf.predict(model, examples)
    correct = sum(p.predicted == e.intent for p, e in zip(preds, examples))
    assert correct / len(examples) > 0.9  # chance is 0.5


def test_predictions_are_well_formed():
    examples = toy_data()
    model, _ = tfidf.train(examples, examples, LABELS)
    preds = tfidf.predict(model, examples)
    check_predictions_cover(preds, examples)  # one answer per example, no extras
    for p in preds:
        assert p.approach == "tfidf_lr"
        assert p.predicted in LABELS
        assert 0.0 <= p.confidence <= 1.0
        assert p.latency_ms > 0


def test_save_and_load_give_identical_predictions(tmp_path):
    examples = toy_data()
    model, _ = tfidf.train(examples, examples, LABELS)
    tfidf.save(model, tmp_path / "m.joblib")
    reloaded = tfidf.load(tmp_path / "m.joblib")
    before = [(p.predicted, p.confidence) for p in tfidf.predict(model, examples)]
    after = [(p.predicted, p.confidence) for p in tfidf.predict(reloaded, examples)]
    assert before == after
