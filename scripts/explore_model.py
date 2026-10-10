"""Step 3.3: look inside DistilBERT, add a fresh classification head, and see it is only guessing.

    python -m scripts.explore_model

No training happens here. We (1) load the model, (2) count its parameters, (3) follow one batch
through it and print every tensor shape, (4) rebuild the head's maths by hand to check we understand
it, (5) measure how good the untrained model is on validation messages.
This is a lesson script, so it prints its findings instead of logging them.
"""

import logging
import math

import torch
import torch.nn.functional as F

from router import log
from router.baselines import distilbert as db
from router.format import render
from router.intents import intent_names
from router.splits import load_split

logger = logging.getLogger("router.explore_model")

N_SHOW = 8  # messages followed through the model in section 3
N_VAL = 500  # validation messages used to measure the untrained model


def encode(tokenizer, texts: list[str], device: str) -> dict:
    """Tokenize a batch (padded to its longest message, cut at MAX_LEN) and move it to the device."""
    batch = tokenizer(texts, padding=True, truncation=True, max_length=db.MAX_LEN, return_tensors="pt")
    return {k: v.to(device) for k, v in batch.items()}


def section_parameters(model) -> None:
    print("\n=== 2. What is inside, and how many learnable numbers? ===")
    body = db.count_parameters(model.distilbert)
    head = db.head_parameters(model)
    total = db.count_parameters(model)
    print(f"  total      {total:>12,}")
    print(f"  body       {body:>12,}  ({body / total:.1%})  pretrained: embeddings + 6 transformer layers")
    print(f"  head       {head:>12,}  ({head / total:.2%})  NEW, random: pre_classifier + classifier")
    emb = db.count_parameters(model.distilbert.embeddings)
    layer = db.count_parameters(model.distilbert.transformer.layer[0])
    print(
        f"    of the body: embeddings {emb:,} (30,522 words x 768 numbers + positions), each of 6 layers {layer:,}"
    )
    pre, cls = model.pre_classifier, model.classifier
    print(
        f"    head part 1: pre_classifier  Linear({pre.in_features} -> {pre.out_features})  = {db.count_parameters(pre):,}"
    )
    print(
        f"    head part 2: classifier      Linear({cls.in_features} -> {cls.out_features})   = {db.count_parameters(cls):,}"
    )
    print("    (a Linear(in -> out) layer has in x out weights + out biases)")


def section_shapes(model, tokenizer, texts: list[str], device: str) -> torch.Tensor:
    """Follow a batch through the model, printing each shape. Returns the logits."""
    print("\n=== 3. One batch through the model, shape by shape ===")
    batch = encode(tokenizer, texts, device)
    n, length = batch["input_ids"].shape
    print(
        f"  input_ids         {tuple(batch['input_ids'].shape)}   = [{n} messages, {length} tokens (longest in this batch)]"
    )
    print(f"  attention_mask    {tuple(batch['attention_mask'].shape)}")
    with torch.no_grad():
        out = model(**batch, output_hidden_states=True)
    hidden = out.hidden_states[-1]
    print(
        f"  hidden states     {tuple(hidden.shape)}   = [{n}, {length}, 768]: one 768-number vector PER TOKEN"
    )
    cls_vec = hidden[:, 0]
    print(
        f"  [CLS] vector      {tuple(cls_vec.shape)}   = the vector at position 0: our summary of each whole message"
    )
    print(
        f"    first 5 numbers of message 0's [CLS] vector: {[round(x, 3) for x in cls_vec[0, :5].tolist()]}"
    )
    print(
        f"  logits            {tuple(out.logits.shape)}   = [{n} messages, 16 intents]: one score per intent"
    )
    print(f"    message 0's logits: {[round(x, 3) for x in out.logits[0].tolist()]}")
    return out.logits, hidden


def section_head_by_hand(model, hidden: torch.Tensor, logits: torch.Tensor, labels: list[str]) -> None:
    print("\n=== 4. The head, rebuilt by hand ===")
    cls_vec = hidden[:, 0]
    with torch.no_grad():
        step1 = model.pre_classifier(cls_vec)  # Linear 768 -> 768
        step2 = torch.relu(step1)  # negative numbers become 0
        by_hand = model.classifier(step2)  # Linear 768 -> 16   (dropout does nothing in eval mode)
    print(f"  [CLS] -> pre_classifier -> ReLU -> classifier   gives logits of shape {tuple(by_hand.shape)}")
    print(f"  identical to the model's own logits? {torch.allclose(by_hand, logits, atol=1e-5)}")

    exp = torch.exp(logits[0])
    probs = exp / exp.sum()  # softmax written out: e^score / sum of all e^scores
    print("\n  Softmax by hand for message 0 (e^score / sum of e^scores):")
    print(
        f"    sum of probabilities = {probs.sum().item():.4f}   matches torch.softmax: {torch.allclose(probs, F.softmax(logits[0], dim=0))}"
    )
    top = torch.topk(probs, 3)
    for p, i in zip(top.values.tolist(), top.indices.tolist(), strict=True):
        print(f"    {labels[i]:<24} {p:6.2%}")
    print(f"    (16 equal guesses would each be {1 / 16:.2%}; the random head is nearly flat, as expected)")


def section_dropout(model, tokenizer, texts: list[str], device: str) -> None:
    print("\n=== 4b. train() vs eval(): dropout ===")
    batch = encode(tokenizer, texts[:2], device)
    model.train()  # dropout ON: randomly zeroes some numbers, to stop the model memorising
    with torch.no_grad():
        a, b = model(**batch).logits, model(**batch).logits
    model.eval()  # dropout OFF: same input always gives the same output
    with torch.no_grad():
        c, d = model(**batch).logits, model(**batch).logits
    print(f"  train mode, same input twice -> same logits? {torch.equal(a, b)}   (dropout makes them differ)")
    print(
        f"  eval  mode, same input twice -> same logits? {torch.equal(c, d)}   (we use eval mode to measure)"
    )


def section_untrained(model, tokenizer, device: str, labels: list[str]) -> None:
    print(f"\n=== 5. How good is the untrained model? ({N_VAL} validation messages) ===")
    val = load_split("val")[:N_VAL]
    texts, gold = [render(e) for e in val], torch.tensor([labels.index(e.intent) for e in val])
    all_logits = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), 50):
            all_logits.append(model(**encode(tokenizer, texts[i : i + 50], device)).logits.cpu())
    logits = torch.cat(all_logits)
    pred = logits.argmax(dim=1)
    accuracy = (pred == gold).float().mean().item()
    loss = F.cross_entropy(logits, gold).item()
    counts = torch.bincount(pred, minlength=len(labels))
    top_class = int(counts.argmax())
    print(f"  accuracy            {accuracy:.1%}   (chance with 16 classes = {1 / 16:.1%})")
    print(
        f"  average loss        {loss:.3f}   (a model that guesses uniformly has ln(16) = {math.log(16):.3f})"
    )
    print(
        f"  classes it ever predicts: {int((counts > 0).sum())} of 16; most common: '{labels[top_class]}' for {counts[top_class].item() / len(pred):.0%} of messages"
    )
    print("  -> before training the head knows nothing; all the learning is still ahead of us.")


def main() -> None:
    log.setup_logging()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    labels = intent_names()
    logger.info("loading %s on %s (first run downloads about 270 MB)", db.MODEL_NAME, device)

    print("\n=== 1. Load the model ===")
    print("  Watch for the next warning: it lists the weights that are NEWLY created (our head).")
    tokenizer = db.load_tokenizer()
    model = db.build_model(labels).to(device)
    model.eval()
    allocated = torch.cuda.memory_allocated() / 1e6 if device == "cuda" else 0
    print(
        f"  class index -> intent: {dict(list(model.config.id2label.items())[:3])} ... (same order as intents.yaml)"
    )
    print(
        f"  GPU memory holding the weights: {allocated:.0f} MB  (about {db.count_parameters(model) * 4 / 1e6:.0f} MB = parameters x 4 bytes)"
    )

    section_parameters(model)
    texts = [render(e) for e in load_split("train")[:N_SHOW]]
    logits, hidden = section_shapes(model, tokenizer, texts, device)
    section_head_by_hand(model, hidden, logits, labels)
    section_dropout(model, tokenizer, texts, device)
    section_untrained(model, tokenizer, device, labels)


if __name__ == "__main__":
    main()
