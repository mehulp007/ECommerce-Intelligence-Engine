"""Ranking metrics for next-week purchase prediction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def average_precision_at_k(predicted: Sequence, actual: set, k: int = 12) -> float:
    """Competition-style average precision, normalized by min(|actual|, k)."""
    if not actual:
        return 0.0
    hits = 0
    total = 0.0
    seen = set()
    for rank, item in enumerate(predicted[:k], start=1):
        if item in actual and item not in seen:
            hits += 1
            total += hits / rank
        seen.add(item)
    return total / min(len(actual), k)


def map_at_k(
    predictions: Mapping[object, Sequence],
    actuals: Mapping[object, set],
    k: int = 12,
) -> float:
    """Mean over customers who bought something; missing predictions score zero."""
    if not actuals:
        raise ValueError("No customers with purchases to evaluate")
    return sum(
        average_precision_at_k(predictions.get(customer, ()), items, k)
        for customer, items in actuals.items()
    ) / len(actuals)


def recall_at_k(
    predictions: Mapping[object, Sequence],
    actuals: Mapping[object, set],
    k: int = 12,
) -> float:
    """Share of purchased (customer, article) pairs that appear in the top k."""
    hits = sum(
        len(set(predictions.get(customer, ())[:k]) & items)
        for customer, items in actuals.items()
    )
    return hits / sum(len(items) for items in actuals.values())
