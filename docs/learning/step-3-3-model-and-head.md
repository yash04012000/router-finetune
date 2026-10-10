# Step 3.3: the pretrained model and a fresh classification head, notes and questions

Plan: [../prds/03-distilbert/3-3-model-and-head.md](../prds/03-distilbert/3-3-model-and-head.md) · Maths: [../math/fine-tuning.md](../math/fine-tuning.md) part 1

## What we did

Loaded DistilBERT, let Hugging Face add a new 16-way classification head, counted every parameter, followed one batch through the model printing each tensor shape,
rebuilt the head's arithmetic by hand to prove we understood it, and measured how good the model is *before* any training.

## Code map

| File | What it does |
|---|---|
| `src/router/baselines/distilbert.py` | Starts here and grows each step. Now: `MODEL_NAME`, `MAX_LEN = 64`, `SEED = 42`, `load_tokenizer()`, **`build_model(labels, seed)`** (body + new head), `count_parameters`, `head_parameters` |
| `scripts/explore_model.py` | Lesson script, five numbered sections: load, count, shapes, head by hand (+ dropout), untrained accuracy |
| `docs/math/fine-tuning.md` | Part 1: Linear layer, ReLU, softmax, cross-entropy, full parameter table, all with checked numbers |
| `tests/test_model_head.py` | 11 tests: the worked maths, the parameter table, label order, shapes, "head = pre_classifier, ReLU, classifier on [CLS]", eval vs train mode, same seed same head, untrained near chance. Skipped offline |

Re-run: `python -m scripts.explore_model` (first run downloads about 270 MB).

## What we saw

- The load report: `classifier.*` and `pre_classifier.*` are **MISSING** from the downloaded file, so they were newly created. That *is* our head. Five `vocab_*` weights are
  **UNEXPECTED** (see questions): leftovers from the pretraining task that we do not use.
- **66,965,776 parameters**: body 66,362,880 (99.1%) + head 602,896 (**0.90%**). Weights alone take 268 MB in float32 (we measured 269 MB on the GPU).
- Shapes for 8 messages, longest 36 tokens: `input_ids (8, 36)` → hidden states `(8, 36, 768)` → `[CLS]` vectors `(8, 768)` → logits `(8, 16)`.
- Head by hand (`pre_classifier` → ReLU → `classifier` on the `[CLS]` vector; the 0.2 dropout between them is off in eval mode) **exactly equals** the model's logits. Softmax by hand equals `torch.softmax`.
- Random-head probabilities are almost flat: top three about 7.2%, 6.9%, 6.7% versus 6.25% for a perfectly flat guess.
- Train mode: the same input twice gives different logits (dropout). Eval mode: identical.
- **Untrained model on 500 validation messages: accuracy 7.6%, loss 2.782**, against chance 6.25% and `ln(16) = 2.773`. It predicted only 4 of the 16 classes, `cash_withdrawal` for 58%.

## Questions and answers

**1. Why is the head initialised randomly, and why can we not skip training it?**
The pretrained file contains the *body* (plus the fill-in-the-blank layers from pretraining), but no classification head: it knows about language, not about *our* 16 intents.
There is nothing to load for the head, so it is created with small random numbers (we measured a standard deviation of 0.02, and biases of 0). We cannot skip training it because random numbers carry no knowledge of which `[CLS]` patterns mean "card_payment_problem":
untrained, it scored 7.6% (guessing). Training adjusts the head so its 16 scores line up with the right intents, and nudges the body so its summary vectors make that easier.

**2. What is the shape of the logits for a batch of 8 messages, and what does each number mean?**
`(8, 16)`: one row per message, one column per intent, in `intents.yaml` order (column 0 = `card_ordering`). Each number is a raw score ("evidence for this intent"). They are not
probabilities: they can be negative and do not add to 1. Softmax turns each row into probabilities. The highest score in a row is the model's answer.

**3. The body has about 66M parameters. Why is fine-tuning it on 9k examples not hopeless?**
Because nearly all the knowledge is already in the body: it was pretrained on a huge amount of text, so it already represents what "card", "declined", "transfer" mean and how they combine.
We are not teaching language, only *which patterns in these summaries map to our 16 labels*. That is a far smaller problem than learning from scratch, so a small number of small updates
(low learning rate) on 9k examples is enough. Training 66M parameters from random starting values on 9k examples *would* be hopeless: it would memorise them.

### Questions that came up while building this

**Why is the head 0.90% of the model, when the plan said about 0.02%?** The plan was wrong, and the code showed it. Hugging Face's DistilBERT head has *two* layers:
`pre_classifier` (768→768, 590,592 parameters) then `classifier` (768→16, 12,304). Only the second is 768→16. Total 602,896, which is 0.90%. Corrected in the plan and
the maths page. Lesson: check the plan's numbers against the code.

**What are the `UNEXPECTED` `vocab_*` weights?** DistilBERT was pretrained on a fill-in-the-blank task ("the card was [MASK] at the shop") with its own prediction layers (`vocab_transform`,
`vocab_layer_norm`, `vocab_projector`). We do classification, not fill-in-the-blank, so we drop them. HF tells you, so you notice if a *needed* layer were dropped by mistake.

**Why take only the `[CLS]` vector and ignore the other tokens' vectors?** Because the model was designed and pretrained to put a summary of the whole sequence at position 0 (it can look at all the other tokens
through attention). The other positions are still used *inside* the body to build that summary. Alternatives exist (average all tokens); `[CLS]` is the standard and what HF does.

**What is dropout, and why `model.train()` versus `model.eval()`?** In train mode, dropout randomly sets some numbers to 0 on each forward pass. This stops the model relying on any single number and fights
memorising (overfitting). When we *measure* or *predict* we want the same input to always give the same answer, so we switch dropout off with `eval()`. Forgetting `eval()` when measuring gives
slightly random scores. We will switch modes deliberately in steps 3.4 and 3.5.

**What does `torch.no_grad()` do?** During training PyTorch records every operation so it can compute gradients afterwards. When only measuring, that recording is wasted memory and time, so `no_grad()` turns it off.
(Step 3.4 shows what the recording is for.)

**Why is the untrained accuracy 7.6% and not exactly 6.25%?** Because "random" here is *consistently* random: this particular random head happens to prefer 4 classes, `cash_withdrawal` most (58%). That class is about
10.7% of validation, so accuracy depends on how well the favoured classes match the data. Over many different random heads it would average near 6.25%. What matters: near chance, and loss near `ln(16)`.

**Why do we set a seed before `from_pretrained`?** The head's random numbers come from PyTorch's random generator. With `torch.manual_seed(42)` first, every run starts from the *same* head, so
experiments are repeatable and comparable (a test checks: same seed, same head; different seed, different head).

**Why does the class order matter so much?** The model outputs "column 7". What does 7 mean? We store `id2label` from `intents.yaml` order inside the model, and use the same order in confusion matrices
and saved predictions. A different order would silently relabel every answer. A test checks it.

## Gotchas

- Hugging Face prints a symlink warning on Windows (harmless) and a long "LOAD REPORT" table. The `MISSING`/`UNEXPECTED` lines are the part to read.
- `hidden_states[-1]` is the output of the last layer (index 0 is the embeddings); the head uses the last one.
- The first model load takes a few seconds even when cached; training scripts should load once, not per batch.

## Where this connects later

Step 3.4 takes exactly this model and does one gradient step on one batch. The initial loss we measured (2.78, about `ln 16`) is the number the 3.4 loss curve must start from. In PRD 4,
the LLM version has the same shape of idea (body + small head) with a much larger body, and LoRA changes *which* parameters we train.
