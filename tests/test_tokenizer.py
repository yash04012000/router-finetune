"""Tests for step 3.2: what the DistilBERT tokenizer does with our messages.

Skipped when transformers is missing, or when the tokenizer has not been downloaded yet
(run `python -m scripts.explore_tokenizer` once), so CI and offline runs stay green.
"""

import pytest

transformers = pytest.importorskip("transformers")

from scripts import explore_tokenizer


@pytest.fixture(scope="module")
def tokenizer():
    try:
        return transformers.AutoTokenizer.from_pretrained(explore_tokenizer.MODEL_NAME, local_files_only=True)
    except OSError:
        pytest.skip("tokenizer not downloaded yet")


def test_special_tokens_wrap_the_message(tokenizer):
    ids = tokenizer("my card")["input_ids"]
    assert ids[0] == tokenizer.cls_token_id == 101
    assert ids[-1] == tokenizer.sep_token_id == 102
    assert tokenizer.convert_ids_to_tokens(ids) == ["[CLS]", "my", "card", "[SEP]"]


def test_it_lowercases_because_the_model_is_uncased(tokenizer):
    assert tokenizer("MY CARD")["input_ids"] == tokenizer("my card")["input_ids"]


def test_padding_makes_a_rectangle_and_the_mask_marks_the_padding(tokenizer):
    batch = tokenizer(["hi", "I need to change my PIN please"], padding=True)
    short_ids, long_ids = batch["input_ids"]
    assert len(short_ids) == len(long_ids)  # rectangle
    n_pad = short_ids.count(tokenizer.pad_token_id)
    assert n_pad > 0 and batch["attention_mask"][0].count(0) == n_pad
    assert 0 not in batch["attention_mask"][1]  # the longest message has no padding


def test_truncation_cuts_to_max_len_but_keeps_the_end_marker(tokenizer):
    ids = tokenizer("one two three four five six seven eight nine ten", truncation=True, max_length=6)[
        "input_ids"
    ]
    assert len(ids) == 6 and ids[-1] == tokenizer.sep_token_id


def test_a_typo_is_split_into_more_pieces_but_never_unknown(tokenizer):
    right = tokenizer.tokenize("received")
    typo = tokenizer.tokenize("recieved")
    assert right == ["received"]
    assert len(typo) > 1 and tokenizer.unk_token not in typo
    assert typo[1].startswith("##")  # '##' = continues the previous piece


def test_length_report_counts_what_each_max_len_would_cut(tokenizer):
    report = explore_tokenizer.length_report(
        tokenizer, ["hi", "my card was declined at the shop", "x " * 100]
    )
    assert report["n_messages"] == 3 and report["max"] == 102  # 100 words + [CLS] + [SEP]
    cut = report["fraction_truncated_at_max_len"]
    assert cut["128"] == 0.0 and cut["16"] == pytest.approx(1 / 3)
    shares = [cut[str(m)] for m in explore_tokenizer.CANDIDATE_MAX_LENS]
    assert shares == sorted(shares, reverse=True)  # a bigger max_len never cuts more
