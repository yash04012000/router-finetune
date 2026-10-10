# Glossary

Plain-words definitions. "First seen" tells you where to find the full explanation. Terms are added as we meet them.

## The task

| Term | Meaning | First seen |
|---|---|---|
| **Intent** | What the customer wants (`card_payment_problem`, `top_up`, ...). There are 16 | PRD 1 |
| **Router** | The thing that reads a message and picks the intent | PRD 1 |
| **Split (train / val / test)** | Train = the model learns from it. Val = we choose settings with it. Test = the final exam, looked at once | PRD 2, [data-splits](../math/data-splits.md) |
| **Frozen split** | Files fingerprinted with SHA-256; loading refuses if they changed, so scores stay comparable | PRD 2 |
| **Gold label** | The correct intent of an example | PRD 1 |

## Metrics

| Term | Meaning | First seen |
|---|---|---|
| **Accuracy** | Fraction answered correctly | [metrics](../math/metrics.md) |
| **Precision / recall / F1** | "When I say X, how often right?" / "Of the real X, how many did I find?" / their harmonic mean | metrics |
| **Macro F1** | Average F1 across classes, each class equal, so small intents count | metrics |
| **Confusion matrix** | Table of true class (rows) vs predicted class (columns) | metrics |
| **p50 / p95 latency** | The typical time / the time 95% of requests beat | metrics |
| **Confidence** | The model's own probability for its answer | TF-IDF |

## Classical model

| Term | Meaning | First seen |
|---|---|---|
| **TF-IDF** | A score per word: high if frequent in this message, low if in every message | [tfidf-logreg](../math/tfidf-logreg.md) |
| **n-gram** | A run of n words (or letters) | tfidf-logreg |
| **Logistic regression** | Learns a weight per (feature, intent); softmax turns scores into probabilities | tfidf-logreg |
| **Softmax** | Turns scores into probabilities that add to 1 | tfidf-logreg |
| **Cross-entropy loss** | `-ln(probability of the right answer)`: small if confident and right, large if confident and wrong | tfidf-logreg |
| **Regularisation (`C`)** | A penalty on big weights so the model does not memorise. Small `C` = stronger penalty | tfidf-logreg |
| **Class weights** | Make mistakes on rare classes count more | tfidf-logreg |

## Hardware and PyTorch (step 3.1)

| Term | Meaning |
|---|---|
| **Tensor** | A grid of numbers: vector, matrix, or a stack of matrices. Described by shape, dtype, device |
| **Shape** | The size of each dimension, for example `(8, 16)` = 8 rows, 16 columns |
| **dtype** | The number format: `float32` (4 bytes), `float16`/`bfloat16` (2 bytes), `int64`... |
| **Device** | Where the tensor lives: `cpu` or `cuda` (the GPU). Operands must be on the same device |
| **CUDA** | NVIDIA's platform for running computation on the GPU. PyTorch must be a CUDA *build* to use it |
| **VRAM** | The GPU's own memory (8 GB on the RTX 4060 Ti). Model, gradients and activations must fit |
| **`synchronize()`** | Wait for the GPU to finish. The GPU runs asynchronously, so timing needs it |
| **Warm-up** | A throw-away first run, because the first call pays one-off start-up costs |
| **bf16** | A 16-bit format with float32's range but less precision; half the memory, no loss scaling needed |

## Language models (step 3.2 onward)

| Term | Meaning |
|---|---|
| **Token** | One piece of text from the tokenizer's vocabulary (a word, part of a word, or punctuation) |
| **Tokenizer** | Cuts text into tokens and maps each to an id. Must be the one the model was trained with |
| **Vocabulary** | The fixed list of pieces. DistilBERT's has 30,522 |
| **WordPiece** | The splitting method BERT-family models use. Unseen words become known smaller pieces |
| **`##`** | "This piece continues the previous one" (`rec ##ie ##ved`) |
| **`input_ids`** | The token ids: what the model actually receives |
| **`[CLS]`** | Special start token (id 101). Its output vector is the message summary the head reads |
| **`[SEP]`** | Special end token (id 102) |
| **`[PAD]`** | Filler (id 0) so a batch is a rectangle |
| **`[UNK]`** | The "unknown" token. WordPiece almost never needs it |
| **`attention_mask`** | 1 for real tokens, 0 for padding, so the model ignores the padding |
| **Padding** | Adding `[PAD]` so all messages in a batch are the same length |
| **Dynamic padding** | Padding each batch only to its own longest message, not to a global maximum |
| **Truncation** | Cutting a message that is longer than `max_len` |
| **`max_len`** | The longest input we allow. We chose 64 (cuts 0.29% of training messages) |
| **Uncased** | The model lowercases everything (`distilbert-base-uncased`) |

## The model (step 3.3)

| Term | Meaning |
|---|---|
| **Pretrained model** | A model already trained by others on huge text. We start from it instead of from nothing |
| **Fine-tuning** | Continuing to train a pretrained model on our small task so it specialises |
| **Body / backbone** | The pretrained part (DistilBERT: embeddings + 6 transformer layers, 66.4M parameters) |
| **Head** | The small new part on top that outputs one score per intent (602,896 parameters, random at the start) |
| **Parameter** | One learnable number (a weight or a bias). Training changes them |
| **Weight / bias** | A `Linear(in -> out)` layer has `in x out` weights and `out` biases |
| **Embedding** | A table that gives each token id a vector of 768 numbers |
| **Transformer layer** | A block that lets every token look at the others (attention) and then transforms them. DistilBERT has 6 |
| **Hidden state** | The 768-number vector for each token after a layer |
| **Attention** | The mechanism by which each token takes information from the other tokens in the message |
| **`[CLS]` vector** | The hidden state at position 0 after the last layer; used as the summary of the message |
| **Logits** | The head's raw scores, one per intent. Any real number; not probabilities |
| **ReLU** | Keeps positive numbers, turns negative numbers into 0 |
| **Softmax** | `e^score / sum of e^scores`: turns logits into probabilities that add to 1 |
| **Dropout** | In training mode, randomly zeroes some numbers to fight memorising. Off in eval mode |
| **`model.train()` / `model.eval()`** | Switch dropout on / off. Use `eval()` whenever you measure or predict |
| **`torch.no_grad()`** | Do not record operations for gradients (saves memory when only measuring) |
| **`id2label` / `label2id`** | The mapping between a class index and its intent name; here, `intents.yaml` order |
| **Seed** | The starting point of random number generation; the same seed gives the same "random" numbers |
| **Chance level** | The accuracy of random guessing: 1/16 = 6.25% for 16 classes |
| **`ln(16)` = 2.773** | The cross-entropy loss of a model that gives all 16 classes equal probability; where training starts |

## Coming in later steps

Gradient, backward pass, optimizer (AdamW), learning rate, batch, epoch, overfitting, mixed precision, calibration, LoRA.
