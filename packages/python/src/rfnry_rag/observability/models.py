from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MetricResult:
    """Aggregated result from a batch metric evaluation."""

    mean: float
    scores: list[float]
