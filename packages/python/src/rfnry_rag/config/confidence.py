from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ConfidenceConfig:
    """Configuration for composite confidence scoring."""

    weights: dict[str, float] = field(
        default_factory=lambda: {
            "retrieval": 0.35,
            "relevance": 0.30,
            "coverage": 0.20,
            "agreement": 0.15,
        }
    )
    thresholds: dict[str, float] = field(
        default_factory=lambda: {
            "high": 0.80,
            "medium": 0.50,
        }
    )

    def __post_init__(self) -> None:
        total = sum(self.weights.values())
        if total <= 0:
            raise ValueError("Sum of weights must be positive")
