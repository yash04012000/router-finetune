# 3.2 — Tokenization

Status: Not started · Depends on: 3.1 · Next: [3.3](3-3-model-and-head.md)

## The one idea

A model cannot read words. A **tokenizer** cuts text into pieces from a fixed vocabulary and replaces
each piece with a number (its id). DistilBERT's vocabulary has 30,522 pieces. Rare or misspelled
words are split into smaller known pieces, so nothing is ever "unknown".

## Goal

See exactly what DistilBERT will receive for our messages, and decide the maximum length (`max_len`)
from real data instead of guessing.

## Concepts you will learn

- WordPiece: why `"declined"` may stay whole but `"recieved"` becomes several pieces (compare with the
  character n-grams in TF-IDF).
- Special tokens: `[CLS]` (start; its output becomes the "summary" of the message), `[SEP]` (end),
  `[PAD]` (filler so a batch is a rectangle).
- `input_ids` and `attention_mask`: the two tensors a model actually takes, and why the mask exists
  (so padding is ignored).
- Truncation and padding, and why `max_len` trades speed and memory against cutting off long messages.
- Lowercasing: `distilbert-base-uncased` lowercases everything.

## What we build

```
scripts/explore_tokenizer.py    prints a few messages as tokens + ids + mask; length statistics
docs/prds/03-distilbert/        (this folder) gets a short note with what we found
```

No training code. The tokenizer is downloaded once from the Hugging Face Hub (about 1 MB) and cached.

## Hands-on

1. Tokenize `"My card was declined at the shop"` and `"I recieved the wrong amout"`. Read the pieces
   and ids side by side.
2. Tokenize a batch of 4 messages of different lengths with padding and print the mask. Spot the zeros.
3. Tokenize all of `train`, print the length distribution (mean, median, p95, p99, max) and what
   fraction would be cut at `max_len` of 32, 64, 128.
4. Choose `max_len` (expected to be small, because these are short single messages). Write down why.

## Done when

- You can read a printed token list and say which special tokens are where.
- A `max_len` is chosen with the evidence (percentage truncated) recorded in the repo.

## Check your understanding

1. Why does the tokenizer return an `attention_mask` as well as `input_ids`?
2. A message is 20 tokens and `max_len` is 16. What happens to it, and what would we lose?
3. Why is a misspelled word less harmful here than for a plain "one id per word" approach?

## Notes

The training pipeline reuses `router.format.render` (PRD 3A) to turn an example into text before it
reaches the tokenizer, so every approach sees identical input.
