from __future__ import annotations

import hashlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from rfnry_rag.reasoning.common.errors import EvaluationError
from rfnry_rag.reasoning.common.language_model import LanguageModelClient, LanguageModelProvider
from rfnry_rag.reasoning.modules.evaluation.metrics import cosine_similarity
from rfnry_rag.reasoning.modules.evaluation.models import (
    EvaluationConfig,
    EvaluationPair,
    EvaluationReport,
    EvaluationResult,
)
from rfnry_rag.reasoning.modules.evaluation.service import EvaluationService


def _lm_client() -> LanguageModelClient:
    return LanguageModelClient(
        provider=LanguageModelProvider(provider="openai", model="gpt-4o-mini", api_key="test-key"),
    )


class _MockEmbeddings:
    model = "test-model"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            h = hashlib.md5(text.encode()).hexdigest()
            vectors.append([int(c, 16) / 15.0 for c in h[:8]])
        return vectors

    async def embedding_dimension(self) -> int:
        return 8


def _mock_judge_result(score: float = 0.75) -> SimpleNamespace:
    return SimpleNamespace(
        overall_score=score,
        reasoning="good but incomplete",
        dimension_scores={"accuracy": 0.8, "completeness": 0.7},
    )


def test_cosine_identical_vectors():
    assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_cosine_orthogonal_vectors():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_opposite_vectors():
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_cosine_zero_vector_returns_zero():
    assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0


def test_cosine_both_zero_vectors():
    assert cosine_similarity([0.0, 0.0], [0.0, 0.0]) == 0.0


async def test_quality_band_high():
    service = EvaluationService(embeddings=_MockEmbeddings())
    result = await service.evaluate(
        EvaluationPair(generated="hello", reference="hello"),
        config=EvaluationConfig(strategy="similarity", high_threshold=0.5),
    )
    assert result.quality_band == "high"


async def test_quality_band_medium():
    service = EvaluationService(embeddings=_MockEmbeddings())
    result = await service.evaluate(
        EvaluationPair(generated="aaa", reference="zzz"),
        config=EvaluationConfig(strategy="similarity", high_threshold=0.99, medium_threshold=0.0),
    )
    assert result.quality_band == "medium"


async def test_quality_band_low():
    service = EvaluationService(embeddings=_MockEmbeddings())
    result = await service.evaluate(
        EvaluationPair(generated="aaa", reference="zzz"),
        config=EvaluationConfig(strategy="similarity", high_threshold=0.99, medium_threshold=0.99),
    )
    assert result.quality_band == "low"


async def test_similarity_requires_embeddings():
    service = EvaluationService()
    with pytest.raises(EvaluationError, match="requires embeddings"):
        await service.evaluate(
            EvaluationPair(generated="a", reference="b"),
            config=EvaluationConfig(strategy="similarity"),
        )


async def test_judge_requires_lm_client():
    service = EvaluationService()
    with pytest.raises(EvaluationError, match="requires lm_client"):
        await service.evaluate(
            EvaluationPair(generated="a", reference="b"),
            config=EvaluationConfig(strategy="judge"),
        )


async def test_evaluate_batch_similarity():
    service = EvaluationService(embeddings=_MockEmbeddings())
    pairs = [
        EvaluationPair(generated="hello", reference="hello"),
        EvaluationPair(generated="foo", reference="bar"),
        EvaluationPair(generated="world", reference="world"),
    ]
    report = await service.evaluate_batch(pairs, config=EvaluationConfig(strategy="similarity"))
    assert isinstance(report, EvaluationReport)
    assert len(report.results) == 3
    assert report.mean_similarity > 0
    assert report.mean_judge_score is None
    total = sum(report.distribution.values())
    assert total == 3


@patch("rfnry_rag.reasoning.modules.evaluation.metrics.b")
async def test_evaluate_batch_combined(mock_b):
    mock_b.JudgeOutput = AsyncMock(return_value=_mock_judge_result())
    service = EvaluationService(embeddings=_MockEmbeddings(), lm_client=_lm_client())
    pairs = [
        EvaluationPair(generated="a", reference="b"),
        EvaluationPair(generated="c", reference="d"),
    ]
    report = await service.evaluate_batch(pairs, config=EvaluationConfig(strategy="combined"))
    assert report.mean_similarity > 0
    assert report.mean_judge_score is not None
    assert report.mean_judge_score == pytest.approx(0.75)


@patch("rfnry_rag.reasoning.modules.evaluation.metrics.b")
async def test_evaluate_batch_judge_only(mock_b):
    mock_b.JudgeOutput = AsyncMock(return_value=_mock_judge_result(score=0.6))
    service = EvaluationService(embeddings=_MockEmbeddings(), lm_client=_lm_client())
    pairs = [EvaluationPair(generated="a", reference="b")]
    report = await service.evaluate_batch(pairs, config=EvaluationConfig(strategy="judge"))
    assert report.mean_similarity == 0.0
    assert report.mean_judge_score == pytest.approx(0.6)


async def test_evaluate_batch_distribution():
    service = EvaluationService(embeddings=_MockEmbeddings())
    pairs = [
        EvaluationPair(generated="same", reference="same"),
        EvaluationPair(generated="different", reference="completely unrelated words"),
    ]
    report = await service.evaluate_batch(
        pairs,
        config=EvaluationConfig(strategy="similarity", high_threshold=0.99, medium_threshold=0.01),
    )
    assert report.distribution["medium"] >= 1


def test_result_to_dict_minimal():
    r = EvaluationResult(score=0.7)
    d = r.to_dict()
    assert d == {"score": 0.7}
    assert "similarity" not in d


def test_result_to_dict_full():
    r = EvaluationResult(
        score=0.85,
        similarity=0.9,
        judge_score=0.85,
        judge_reasoning="good",
        dimension_scores={"accuracy": 0.9},
        quality_band="high",
    )
    d = r.to_dict()
    assert d["similarity"] == 0.9
    assert d["judge_score"] == 0.85
    assert d["quality_band"] == "high"
    assert d["dimension_scores"]["accuracy"] == 0.9


def test_config_invalid_thresholds():
    with pytest.raises(ValueError, match="Thresholds"):
        EvaluationConfig(high_threshold=0.3, medium_threshold=0.5)


def test_config_concurrency_below_1():
    with pytest.raises(ValueError, match="concurrency"):
        EvaluationConfig(concurrency=0)
