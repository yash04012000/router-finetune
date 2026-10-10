# 3.3 — The pretrained model and a fresh classification head

Status: Done · Depends on: 3.2 · Next: [3.4](3-4-one-training-step.md)

## The one idea

**Fine-tuning = take a model that already understands language (pretrained *body*) and teach it our
task by adding a small new layer on top (the *head*) and training.** The body was trained by others on
a huge amount of text. We only need to nudge it, which is why 9k examples are enough.

## Goal

Load DistilBERT, push one batch through it, look at every tensor shape on the way, and observe that
before any training the model is no better than guessing.

## Concepts you will learn

- What is inside DistilBERT: 6 transformer layers, hidden size 768, 66M parameters (BERT-base has 12 layers;
  DistilBERT is a distilled, smaller copy).
- Hidden states: one 768-number vector per token. We take the vector at the `[CLS]` position as the
  message's summary.
- The head: two layers, `Linear(768 -> 768)`, ReLU, dropout, then `Linear(768 -> 16)`, that turn the summary into 16 scores (**logits**), one per
  intent. Softmax turns logits into probabilities (same softmax as in the TF-IDF maths page).
- Why a brand-new head starts with random weights, so predictions are random.
- Parameter counting: how many are in the body vs the head (the head is about 0.90% of the model).

## What we build

```
scripts/explore_model.py     loads the model + tokenizer, runs one batch, prints shapes, parameter counts,
                             and the accuracy of the untrained model on a slice of val
docs/math/fine-tuning.md     part 1: logits, softmax, what the head computes (formulas + worked example)
```

## Hands-on

1. Load `DistilBertForSequenceClassification` with `num_labels=16` and our label list. Read the warning
   it prints ("some weights are newly initialized"); that is the head, and it is exactly the point.
2. Run a batch of 8 messages. Print `input_ids` `[8, L]`, hidden states `[8, L, 768]`, logits `[8, 16]`.
3. Apply softmax by hand and compare with the library; they must match.
4. Count parameters per part. Measure accuracy of the untrained model on 500 val messages. Expect
   roughly 1/16 (6%), i.e. chance. This number is the baseline for "did training do anything?".
5. Move the model to the GPU and check VRAM used.

## Done when

- You can draw the pipeline `text -> ids -> body -> [CLS] vector -> head -> 16 logits -> probabilities`.
- Untrained accuracy is near chance and we know why.

## Result

- 66,965,776 parameters: body 66,362,880 (99.1%) + head 602,896 (**0.90%**; the first plan said "about 0.02%", which was wrong because the head has two layers).
- Shapes for 8 messages x 36 tokens: ids (8, 36), hidden (8, 36, 768), `[CLS]` (8, 768), logits (8, 16). The head rebuilt by hand matches the model exactly.
- Untrained, on 500 validation messages: accuracy **7.6%** (chance 6.25%), loss **2.782** (`ln 16` = 2.773), only 4 of 16 classes ever predicted.
- **Lab tab** (UI): tokenizer view (3.2) and untrained-model view (3.3): shapes, [CLS] strip, logits and probabilities, parameter table. Backend `src/router/lab.py`, tests `tests/test_lab.py`.
- 11 tests in `tests/test_model_head.py`. Notes: [docs/learning/step-3-3-model-and-head.md](../../learning/step-3-3-model-and-head.md).

## Check your understanding

1. Why is the head initialised randomly, and why can we not skip training it?
2. What is the shape of the logits for a batch of 8 messages, and what does each number mean?
3. The body has about 66M parameters. Why is fine-tuning it on 9k examples not hopeless?
