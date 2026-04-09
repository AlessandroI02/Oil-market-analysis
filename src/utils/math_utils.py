from __future__ import annotations

from typing import Iterable


def safe_weighted_average(values: Iterable[float], weights: Iterable[float]) -> float:
    values_list = list(values)
    weights_list = list(weights)
    if not values_list or not weights_list or len(values_list) != len(weights_list):
        return float("nan")
    denom = sum(weights_list)
    if denom == 0:
        return float("nan")
    return float(sum(v * w for v, w in zip(values_list, weights_list)) / denom)


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(weights.values())
    if total <= 0:
        return {k: 0.0 for k in weights}
    return {k: v / total for k, v in weights.items()}
