"""Step 3.2: see what DistilBERT actually receives for our messages, and choose max_len.

    python -m scripts.explore_tokenizer

Nothing is trained here. We load the tokenizer (downloaded once, about 1 MB, then cached), look at a few
messages as tokens and ids, look at padding, and measure how long our real messages are.
This is a lesson script, so it prints its findings instead of logging them.
"""

import json
import logging

import numpy as np
from transformers import AutoTokenizer

from router import log
from router.format import render
from router.intents import REPO_ROOT
from router.splits import load_split

logger = logging.getLogger("router.explore_tokenizer")

MODEL_NAME = "distilbert-base-uncased"
CANDIDATE_MAX_LENS = [16, 24, 32, 48, 64, 128]
OUT_FILE = REPO_ROOT / "results" / "training" / "tokenizer_lengths.json"


def show_tokens(tokenizer, text: str) -> None:
    """One message: the pieces, and the ids the model actually receives."""
    encoded = tokenizer(text)  # adds [CLS] at the start and [SEP] at the end by itself
    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])
    print(f'\n  text    : "{text}"')
    print(f"  tokens  : {tokens}")
    print(f"  ids     : {encoded['input_ids']}")
    print(f"  length  : {len(tokens)} tokens ({len(text.split())} words)")


def show_padding(tokenizer, texts: list[str]) -> None:
    """A batch must be a rectangle, so shorter messages are filled with [PAD]; the mask marks real tokens."""
    batch = tokenizer(texts, padding=True)
    print("\n  A batch of 4 messages, padded to the longest:")
    for text, ids, mask in zip(texts, batch["input_ids"], batch["attention_mask"], strict=True):
        print(f"    ids  {ids}")
        print(f"    mask {mask}   <- 1 = real token, 0 = padding   ({text!r})")


def length_report(tokenizer, texts: list[str]) -> dict:
    """Token count of every training message (with [CLS] and [SEP]), and how many a given max_len would cut."""
    lengths = np.array([len(ids) for ids in tokenizer(texts)["input_ids"]])
    truncated = {str(m): float((lengths > m).mean()) for m in CANDIDATE_MAX_LENS}
    return {
        "n_messages": len(lengths),
        "mean": float(lengths.mean()),
        "median": float(np.median(lengths)),
        "p95": float(np.percentile(lengths, 95)),
        "p99": float(np.percentile(lengths, 99)),
        "max": int(lengths.max()),
        "fraction_truncated_at_max_len": truncated,
        "lengths": lengths,
    }


def print_histogram(lengths: np.ndarray, bin_width: int = 4) -> None:
    print("\n  How long are our messages? (tokens, one # is about 1% of messages)")
    top = int(lengths.max())
    for start in range(0, top + bin_width, bin_width):
        share = ((lengths >= start) & (lengths < start + bin_width)).mean()
        if share > 0:
            print(f"    {start:>3}-{start + bin_width - 1:<3} {'#' * round(share * 100):<40} {share:5.1%}")


def main() -> None:
    log.setup_logging()
    logger.info("loading tokenizer '%s' (first run downloads about 1 MB)", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    logger.info(
        "vocabulary size: %d pieces; special tokens: %s", tokenizer.vocab_size, tokenizer.all_special_tokens
    )

    print("\n=== 1. Messages as tokens and ids ===")
    show_tokens(tokenizer, "My card was declined at the shop")
    print("\n  Now the same idea with a typo. '##' means 'continues the previous piece':")
    show_tokens(tokenizer, "I received the wrong amount")
    show_tokens(tokenizer, "I recieved the wrong amout")

    print("\n=== 2. Padding and the attention mask ===")
    train = load_split("train")
    show_padding(tokenizer, [render(e) for e in train[:4]])

    print("\n=== 3. How long are our messages? ===")
    report = length_report(tokenizer, [render(e) for e in train])
    print(f"  {report['n_messages']} training messages, tokens incl. [CLS] and [SEP]:")
    print(
        f"  mean {report['mean']:.1f} | median {report['median']:.0f} | p95 {report['p95']:.0f} "
        f"| p99 {report['p99']:.0f} | max {report['max']}"
    )
    print_histogram(report["lengths"])
    print("\n  What each max_len would cut off (share of messages longer than max_len):")
    for m, share in report["fraction_truncated_at_max_len"].items():
        print(f"    max_len {m:>3}: {share:6.2%} truncated")

    del report["lengths"]  # keep the saved file small
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps({"model": MODEL_NAME, **report}, indent=2) + "\n", encoding="utf-8")
    logger.info("saved %s", OUT_FILE)


if __name__ == "__main__":
    main()
