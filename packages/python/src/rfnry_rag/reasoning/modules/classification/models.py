from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass
class CategoryDefinition:
    """A category for classification."""

    name: str
    description: str
    examples: list[str] | None = None


@dataclass
class Classification:
    """Result of a classification operation."""

    category: str
    confidence: float
    strategy_used: Literal["llm", "knn", "hybrid_knn", "hybrid_llm_escalation"]
    reasoning: str | None = None
    runner_up: str | None = None
    runner_up_confidence: float | None = None
    vote_distribution: dict[str, int] | None = None
    evidence: list[dict[str, Any]] | None = None
    metadata: dict[str, Any] | None = None
    needs_review: bool = False

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "category": self.category,
            "confidence": self.confidence,
            "strategy_used": self.strategy_used,
        }
        if self.reasoning is not None:
            d["reasoning"] = self.reasoning
        if self.runner_up is not None:
            d["runner_up"] = self.runner_up
        if self.runner_up_confidence is not None:
            d["runner_up_confidence"] = self.runner_up_confidence
        if self.vote_distribution is not None:
            d["vote_distribution"] = self.vote_distribution
        if self.evidence is not None:
            d["evidence"] = self.evidence
        return d


@dataclass
class ClassificationConfig:
    """Configuration for classification operations."""

    strategy: Literal["llm", "hybrid"] = "llm"
    escalation_threshold: float = 0.7
    top_k: int = 10
    concurrency: int = 10
    knn_knowledge_id: str | None = None
    knn_label_field: str = "category"
    max_text_length: int = 2000
    low_confidence_threshold: float | None = None

    def __post_init__(self) -> None:
        if self.strategy not in ("llm", "hybrid"):
            raise ValueError(f"Unknown strategy: {self.strategy}. Must be 'llm' or 'hybrid'.")
        if not 0.0 <= self.escalation_threshold <= 1.0:
            raise ValueError("escalation_threshold must be between 0.0 and 1.0")
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")
        if self.low_confidence_threshold is not None and not 0.0 <= self.low_confidence_threshold <= 1.0:
            raise ValueError("low_confidence_threshold must be between 0.0 and 1.0")


@dataclass
class ClassificationSetDefinition:
    """Named group of categories for multi-classification."""

    name: str
    categories: list[CategoryDefinition]


@dataclass
class ClassificationSetResult:
    """Result of classifying against multiple category sets."""

    classifications: dict[str, Classification]
