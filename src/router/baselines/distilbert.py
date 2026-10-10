"""Baseline 2: DistilBERT fine-tuned to pick one of our 16 intents.

This file grows one step at a time (see docs/prds/03-distilbert/):
    3.3  build the model: pretrained body + a fresh classification head   <- here
    3.4  one training step by hand
    3.5  the full training loop, evaluation, saving
    ...
"""

import logging

import torch
from transformers import AutoTokenizer, DistilBertForSequenceClassification

logger = logging.getLogger("router.distilbert")

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 64  # chosen in step 3.2: cuts only 0.29% of training messages
SEED = 42


def load_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def build_model(labels: list[str], seed: int = SEED) -> DistilBertForSequenceClassification:
    """The pretrained DistilBERT body plus a NEW, randomly initialised classification head.

    `labels` is the intent list in intents.yaml order: class index i means labels[i], everywhere
    (the confusion matrix, the logits, the saved model). Changing the order would silently scramble answers.

    The body's weights are loaded from the Hugging Face Hub (about 270 MB, cached after the first time).
    The head has no pretrained weights, so transformers creates it with random numbers; we set the
    seed first so that "random" is the same every run.
    """
    torch.manual_seed(seed)
    return DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(labels),
        id2label=dict(enumerate(labels)),
        label2id={name: i for i, name in enumerate(labels)},
    )


def count_parameters(module: torch.nn.Module) -> int:
    """Number of learnable numbers (weights and biases) in a module."""
    return sum(p.numel() for p in module.parameters())


def head_parameters(model: DistilBertForSequenceClassification) -> int:
    """Parameters that are NOT part of the pretrained body: the new classification head."""
    return count_parameters(model.pre_classifier) + count_parameters(model.classifier)
