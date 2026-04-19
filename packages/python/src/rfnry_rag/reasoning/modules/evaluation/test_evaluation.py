from __future__ import annotations

import hashlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from rfnry_rag.reasoning.common.language_model import LanguageModelClient, LanguageModelProvider
from rfnry_rag.reasoning.modules.evaluation.models import (
    EvaluationConfig,
    EvaluationDimensionDefinition,
    EvaluationPair,
    EvaluationResult,
)
from rfnry_rag.reasoning.modules.evaluation.service import EvaluationService


def _lm_client() -> LanguageModelClient:
    return LanguageModelClient(
        provider=LanguageModelProvider(provider="openai", model="gpt-4o-mini", api_key="test-key"),
    )


class _MockEmbeddings:
    """Deterministic mock embeddings for testing."""

    model = "test-model"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            h = hashlib.md5(text.encode()).hexdigest()
            vectors.append([int(c, 16) / 15.0 for c in h[:8]])
        return vectors

    async def embedding_dimension(self) -> int:
        return 8


def _mock_judge_result() -> SimpleNamespace:
    return SimpleNamespace(
        overall_score=0.75,
        reasoning="good but incomplete",
        dimension_scores={"accuracy": 0.8, "completeness": 0.7},
    )


async def test_evaluate_similarity_only():
    """Similarity strategy returns embedding-based score without LLM."""
    service = EvaluationService(embeddings=_MockEmbeddings())
    result = await service.evaluate(
        EvaluationPair(generated="hello world", reference="hello world"),
        config=EvaluationConfig(strategy="similarity"),
    )
    assert isinstance(result, EvaluationResult)
    assert result.similarity is not None
    assert result.score == result.similarity
    assert result.judge_score is None
    assert result.quality_band is not None


@patch("rfnry_rag.reasoning.modules.evaluation.metrics.b")
async def test_evaluate_judge_only(mock_b):
    """Judge strategy returns LLM score without embedding similarity."""
    mock_b.JudgeOutput = AsyncMock(return_value=_mock_judge_result())
    service = EvaluationService(embeddings=_MockEmbeddings(), lm_client=_lm_client())
    result = await service.evaluate(
        EvaluationPair(generated="answer", reference="reference"),
        config=EvaluationConfig(
            strategy="judge",
            dimensions=[
                EvaluationDimensionDefinition("accuracy", "factual correctness"),
                EvaluationDimensionDefinition("completeness", "covers all points"),
            ],
        ),
    )
    assert result.judge_score == pytest.approx(0.75)
    assert result.score == result.judge_score
    assert result.similarity is None
    assert result.dimension_scores == {"accuracy": 0.8, "completeness": 0.7}


@patch("rfnry_rag.reasoning.modules.evaluation.metrics.b")
async def test_evaluate_combined(mock_b):
    """Combined strategy returns both similarity and judge scores."""
    mock_b.JudgeOutput = AsyncMock(return_value=_mock_judge_result())
    service = EvaluationService(embeddings=_MockEmbeddings(), lm_client=_lm_client())
    result = await service.evaluate(
        EvaluationPair(generated="answer", reference="reference"),
        config=EvaluationConfig(strategy="combined"),
    )
    assert result.similarity is not None
    assert result.judge_score is not None
    assert result.score == result.judge_score


async def test_evaluate_quality_band_high():
    """Score above high_threshold produces quality_band='high'."""
    service = EvaluationService(embeddings=_MockEmbeddings())
    result = await service.evaluate(
        EvaluationPair(generated="hello", reference="hello"),
        config=EvaluationConfig(strategy="similarity", high_threshold=0.5),
    )
    assert result.quality_band == "high"


def test_evaluation_config_strategy_validation():
    """Invalid strategy raises ValueError."""
    with pytest.raises(ValueError, match="Unknown strategy"):
        EvaluationConfig(strategy="invalid")
