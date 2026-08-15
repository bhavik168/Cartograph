"""Per-model price table, in USD per 1,000 tokens.

# verify against current provider pricing

Prices change and this table is a plain dict on purpose: edit it. The USD
figures in every audit report are only as correct as what you maintain here.
Cartographer does not claim authoritative rates and does not fetch them.

Cached input tokens are billed at ``CACHE_READ_MULTIPLIER`` of the input rate.
"""

from __future__ import annotations

# model prefix -> (input_usd_per_1k, output_usd_per_1k)
PRICES: dict[str, tuple[float, float]] = {
    # Anthropic
    "claude-opus-4": (0.015, 0.075),
    "claude-sonnet-4": (0.003, 0.015),
    "claude-haiku-4": (0.001, 0.005),
    "claude-3-5-sonnet": (0.003, 0.015),
    "claude-3-5-haiku": (0.0008, 0.004),
    # OpenAI
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4o": (0.0025, 0.01),
    "gpt-4.1-mini": (0.0004, 0.0016),
    "gpt-4.1": (0.002, 0.008),
}

CACHE_READ_MULTIPLIER = 0.1

# Used when a model id matches nothing above. Zero, not a guess: a silent
# wrong price is worse than an obvious missing one, and unpriced models are
# surfaced in the audit report.
UNKNOWN_PRICE = (0.0, 0.0)


def rates_for(model: str) -> tuple[float, float]:
    """Longest-prefix match so dated model ids resolve to their family rate."""
    match = ""
    for prefix in PRICES:
        if model.startswith(prefix) and len(prefix) > len(match):
            match = prefix
    return PRICES[match] if match else UNKNOWN_PRICE


def is_priced(model: str) -> bool:
    return rates_for(model) != UNKNOWN_PRICE


def cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0,
) -> float:
    """Estimated USD for one call. Cached input is billed at the reduced rate."""
    in_rate, out_rate = rates_for(model)
    fresh_input = max(input_tokens - cached_input_tokens, 0)
    return (
        fresh_input * in_rate
        + cached_input_tokens * in_rate * CACHE_READ_MULTIPLIER
        + output_tokens * out_rate
    ) / 1000.0
