from __future__ import annotations

from dataclasses import dataclass, field

from rfnry_rag.generation.models import RelevanceResult
from rfnry_rag.models import RetrievedChunk


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


@dataclass
class ConfidenceScore:
    """Composite confidence score with breakdown."""

    level: str
    value: float
    breakdown: dict[str, float]
    reasons: list[str]


class ConfidenceScorer:
    """Multi-signal confidence scorer for RAG pipeline outputs."""

    def __init__(self, config: ConfidenceConfig | None = None) -> None:
        self._config = config or ConfidenceConfig()

    def score(
        self,
        chunks: list[RetrievedChunk],
        relevance_result: RelevanceResult | None = None,
        expected_source_types: list[str] | None = None,
    ) -> ConfidenceScore:
        """Compute a weighted composite confidence score from multiple signals."""
        if not chunks:
            return ConfidenceScore(
                level="low",
                value=0.0,
                breakdown={"retrieval": 0.0, "relevance": 0.0, "coverage": 0.0, "agreement": 0.0},
                reasons=["No retrieval results available"],
            )

        signals = {}
        reasons: list[str] = []

        retrieval_score = max(c.score for c in chunks)
        signals["retrieval"] = retrieval_score
        if retrieval_score >= 0.8:
            reasons.append(f"Strong retrieval match ({retrieval_score:.2f})")
        elif retrieval_score < 0.5:
            reasons.append(f"Weak retrieval match ({retrieval_score:.2f})")

        if relevance_result is not None:
            signals["relevance"] = relevance_result.confidence
            if relevance_result.confidence >= 0.8:
                reasons.append("Relevance gate confirms high relevance")
            elif relevance_result.confidence < 0.5:
                reasons.append("Relevance gate indicates low relevance")
        else:
            signals["relevance"] = retrieval_score

        if expected_source_types:
            present = {c.source_type for c in chunks if c.source_type}
            covered = sum(1 for st in expected_source_types if st in present)
            coverage = covered / len(expected_source_types)
            signals["coverage"] = coverage
            missing = [st for st in expected_source_types if st not in present]
            if missing:
                reasons.append(f"Missing source types: {', '.join(missing)}")
            else:
                reasons.append("All expected source types represented")
        else:
            signals["coverage"] = 1.0

        if len(chunks) >= 2:
            scores = [c.score for c in chunks]
            spread = max(scores) - min(scores)
            agreement = 1.0 - min(spread, 1.0)
            signals["agreement"] = agreement
            if agreement < 0.5:
                reasons.append("Retrieved results show high score variance")
        else:
            signals["agreement"] = 1.0

        weights = self._config.weights
        total_weight = sum(weights.get(k, 0) for k in signals)
        composite = 0.0 if total_weight == 0 else sum(signals[k] * weights.get(k, 0) for k in signals) / total_weight

        thresholds = self._config.thresholds
        if composite >= thresholds.get("high", 0.80):
            level = "high"
        elif composite >= thresholds.get("medium", 0.50):
            level = "medium"
        else:
            level = "low"

        return ConfidenceScore(
            level=level,
            value=round(composite, 4),
            breakdown={k: round(v, 4) for k, v in signals.items()},
            reasons=reasons,
        )
