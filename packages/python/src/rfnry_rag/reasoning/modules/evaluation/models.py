from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class EvaluationPair:
    """A single pair of generated and reference texts for evaluation."""

    generated: str
    reference: str
    context: str | None = None


@dataclass
class EvaluationDimensionDefinition:
    """Consumer defines quality scoring axes."""

    name: str
    description: str


@dataclass
class EvaluationResult:
    """Result of evaluating a single generated-vs-reference pair."""

    score: float
    similarity: float | None = None
    judge_score: float | None = None
    judge_reasoning: str | None = None
    dimension_scores: dict[str, float] | None = None
    quality_band: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"score": self.score}
        if self.similarity is not None:
            d["similarity"] = self.similarity
        if self.judge_score is not None:
            d["judge_score"] = self.judge_score
        if self.judge_reasoning is not None:
            d["judge_reasoning"] = self.judge_reasoning
        if self.dimension_scores is not None:
            d["dimension_scores"] = self.dimension_scores
        if self.quality_band is not None:
            d["quality_band"] = self.quality_band
        return d


@dataclass
class EvaluationConfig:
    """Configuration for evaluation operations."""

    strategy: Literal["similarity", "judge", "combined"] = "similarity"
    dimensions: list[EvaluationDimensionDefinition] = field(default_factory=list)
    concurrency: int = 10
    high_threshold: float = 0.8
    medium_threshold: float = 0.5
    max_text_length: int = 3000

    def __post_init__(self) -> None:
        if self.strategy not in ("similarity", "judge", "combined"):
            raise ValueError(f"Unknown strategy: {self.strategy}. Must be 'similarity', 'judge', or 'combined'.")
        if not 0.0 <= self.medium_threshold <= self.high_threshold <= 1.0:
            raise ValueError(
                f"Thresholds must satisfy 0 <= medium ({self.medium_threshold}) <= high ({self.high_threshold}) <= 1"
            )
        if self.concurrency < 1:
            raise ValueError("concurrency must be >= 1")


@dataclass
class EvaluationReport:
    """Aggregate report from batch evaluation."""

    results: list[EvaluationResult]
    mean_similarity: float = 0.0
    mean_judge_score: float | None = None
    distribution: dict[str, int] = field(default_factory=lambda: {"high": 0, "medium": 0, "low": 0})
