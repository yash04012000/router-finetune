# Step 3.2: tokenization, notes and questions

Plan: [../prds/03-distilbert/3-2-tokenizer.md](../prds/03-distilbert/3-2-tokenizer.md)

## What we did

Loaded DistilBERT's tokenizer, looked at our messages as tokens and ids, looked at padding and the attention mask, measured how long
our real messages are, and chose `max_len` from the evidence.

## Code map

| File | What it does |
|---|---|
| `scripts/explore_tokenizer.py` | A lesson script that prints what it finds (so it uses `print`, on purpose) |
| | `show_tokens(tokenizer, text)`: pieces and ids of one message |
| | `show_padding(tokenizer, texts)`: a padded batch with its `attention_mask` |
| | `length_report(tokenizer, texts)`: token counts, percentiles, and share truncated at each candidate `max_len` |
| | `print_histogram(lengths)`: a text histogram |
| `results/training/tokenizer_lengths.json` | The saved numbers behind the `max_len` decision |
| `tests/test_tokenizer.py` | Special tokens, lowercasing, padding/mask, truncation keeps `[SEP]`, typo splitting, truncation shares. Skipped offline |

Re-run: `python -m scripts.explore_tokenizer` (first run downloads about 1 MB, then it is cached in `~/.cache/huggingface`).

## What we saw

```
"My card was declined at the shop"
[CLS] my card was declined at the shop [SEP]
[101, 2026, 4003, 2001, 6430, 2012, 1996, 4497, 102]

"I received the wrong amount"  ->  [CLS] i received the wrong amount [SEP]               7 tokens
"I recieved the wrong amout"   ->  [CLS] i rec ##ie ##ved the wrong am ##out [SEP]      10 tokens
```

- Everything is lowercased (`distilbert-base-uncased`). `[CLS]` is id 101, `[SEP]` is 102, `[PAD]` is 0.
- Token lengths over 9,371 training messages: mean 16.0, median 13, p95 36, p99 53, max 98.
- Share cut at `max_len` 16 / 24 / 32 / 48 / **64** / 128: 27.8% / 11.8% / 6.7% / 1.4% / **0.29%** / 0%. **We chose 64.**
- In a padded batch of 4, one 36-token message forced 32 `[PAD]` tokens onto a 7-token one.

## Questions and answers

**1. Why does the tokenizer return an `attention_mask` as well as `input_ids`?**
A batch must be a rectangle, so shorter messages are filled with `[PAD]`. The model's attention layers let every token look at every other token; if the
filler were treated as real words it would blur the meaning of the real tokens. The mask (1 = real token, 0 = padding) tells the model to ignore the padded
positions (inside attention their scores are pushed to minus infinity before the softmax, so they get zero weight). The result should not depend on how
much padding a message happened to receive.

**2. A message is 20 tokens and `max_len` is 16. What happens, and what do we lose?**
It is truncated to 16 tokens *including* `[CLS]` and `[SEP]`: `[CLS]`, the first 14 real tokens, `[SEP]`. The last 4 real tokens are dropped (the tokenizer cuts from
the right by default; our test checks `[SEP]` stays at the end). We lose the end of the message, which can carry the key information ("...Is it fraud?").
That is why `max_len` is chosen from measured lengths: at 64, only 0.29% of messages lose anything.

**3. Why is a misspelled word less harmful here than with a plain "one id per word" scheme?**
With one id per word, an unseen word like `amout` becomes a single `[UNK]` id: every unknown word looks the same and no information survives. WordPiece splits it
into known pieces (`am ##out`) that carry some meaning for the pretrained model. It is less harmful, not harmless: `rec ##ie ##ved` is not the same signal as
`received`, and the neighbouring words have to help. (Compare TF-IDF, which handled typos with character n-grams for a similar reason.)

**Why does a bigger `max_len` cost nothing for short messages?** We pad each batch only to its *own* longest message (dynamic padding, step 3.5).
`max_len` is just a ceiling for unusually long messages. It matters for memory in the worst batch, not for the typical one.

**Why must we use DistilBERT's own tokenizer and not build our own?** The pretrained model learned what each id *means* using this exact vocabulary
(id 4003 is "card" to it). A different vocabulary would feed it meaningless numbers.

**Why does the tokenizer add `[CLS]` and `[SEP]`?** The model was pretrained with them. `[CLS]` is at position 0, and its output vector becomes the
"summary of the whole message" that our classification head reads in step 3.3. `[SEP]` marks the end (and separates two sentences in other tasks).

**What is `##`?** "This piece continues the previous piece without a space." `rec ##ie ##ved` reassembles into `recieved`.

## Gotchas

- Hugging Face prints harmless warnings: unauthenticated requests (set `HF_TOKEN` for higher limits) and no symlink support on Windows (the cache just
  uses a bit more disk). Neither affects results.
- `tokenizer(...)` adds the special tokens for you; do not add `[CLS]` by hand.
- The tokenizer step is deterministic: same text in, same ids out. There is no randomness here.

## Where this connects later

Step 3.3 feeds these ids into the model. Step 3.5's training loop pads each batch dynamically. PRD 4 (Qwen) has a different tokenizer with its own
quirks (for example, no pad token by default), so understanding this step makes that one easier.
