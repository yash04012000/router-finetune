"""Turn an Example into the single string a model reads.

Every non-LLM approach (TF-IDF, DistilBERT, LoRA) calls this one function, so they all see
the same input. The public data is single-message, so today this just returns the message.
If an example ever has earlier turns, they are kept as context with a role tag in front.
"""

from router.schema import Example


def render(example: Example) -> str:
    if len(example.turns) == 1:
        return example.text
    return " ".join(f"[{t.role}] {t.text}" for t in example.turns)
