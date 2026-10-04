"""Cost per 1,000 routing decisions. See docs/math/cost.md for the formulas.

Two kinds of model, two ways to be charged:
  API model     -> you pay per token, so cost = tokens used x price per token
  Self-hosted   -> you pay per hour of machine, so cost = hourly price / decisions per hour
"""

from pathlib import Path

import yaml

from router.schema import Prediction

PRICING_FILE = Path(__file__).resolve().parents[2] / "config" / "pricing.yaml"


def load_pricing(path: Path = PRICING_FILE) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def api_cost_per_1k(preds: list[Prediction], model: str, pricing: dict) -> float:
    """Average (input_tokens x input price + output_tokens x output price) x 1000 decisions."""
    if model not in pricing["api"]:
        raise KeyError(f"No price for '{model}' in pricing.yaml - refusing to assume it is free")
    price = pricing["api"][model]
    total = 0.0
    for p in preds:
        if p.input_tokens is None or p.output_tokens is None:
            raise ValueError(f"Prediction {p.example_id} has no token counts")
        total += p.input_tokens * price["input_per_mtok"] / 1e6
        total += p.output_tokens * price["output_per_mtok"] / 1e6
    return total / len(preds) * 1000


def self_hosted_cost_per_1k(decisions_per_second: float, hour_price_usd: float) -> float:
    """hour price / decisions per hour x 1000.

    decisions_per_second should be the BATCHED throughput - how fast a busy server really
    goes - not the one-at-a-time speed, which would make the model look needlessly expensive.
    """
    decisions_per_hour = decisions_per_second * 3600
    return hour_price_usd / decisions_per_hour * 1000


def hybrid_cost_per_1k(small_cost_per_1k: float, large_cost_per_1k: float, fallback_rate: float) -> float:
    """The small model always runs; the large model runs only on the fraction we fall back on."""
    return small_cost_per_1k + fallback_rate * large_cost_per_1k
