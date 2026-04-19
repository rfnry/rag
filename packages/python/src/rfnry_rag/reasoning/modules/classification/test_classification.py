from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from rfnry_rag.reasoning.common.errors import ClassificationError
from rfnry_rag.reasoning.common.language_model import LanguageModelClient, LanguageModelProvider
from rfnry_rag.reasoning.modules.classification.models import (
    CategoryDefinition,
    ClassificationConfig,
    ClassificationSetDefinition,
    ClassificationSetResult,
)
from rfnry_rag.reasoning.modules.classification.service import ClassificationService


def _lm_client() -> LanguageModelClient:
    return LanguageModelClient(
        provider=LanguageModelProvider(provider="openai", model="gpt-4o-mini", api_key="test-key"),
    )


def _categories() -> list[CategoryDefinition]:
    return [
        CategoryDefinition("billing", "billing related"),
        CategoryDefinition("shipping", "shipping related"),
        CategoryDefinition("product", "product related"),
    ]


def _mock_classify_result(category: str = "shipping", confidence: float = 0.9) -> SimpleNamespace:
    return SimpleNamespace(
        category=category,
        confidence=confidence,
        reasoning="test reasoning",
        runner_up="billing",
        runner_up_confidence=0.1,
    )


def _mock_sets_result() -> SimpleNamespace:
    return SimpleNamespace(
        classifications=[
            SimpleNamespace(
                set_name="topic",
                category="shipping",
                confidence=0.9,
                reasoning="about delivery",
                runner_up="billing",
                runner_up_confidence=0.1,
            ),
            SimpleNamespace(
                set_name="channel",
                category="complaint",
                confidence=0.85,
                reasoning="frustrated tone",
                runner_up=None,
                runner_up_confidence=None,
            ),
        ]
    )


@patch("rfnry_rag.reasoning.modules.classification.strategies.b")
async def test_classify_sets(mock_b):
    """Multi-set classification returns one result per set."""
    mock_b.ClassifyTextSets = AsyncMock(return_value=_mock_sets_result())
    service = ClassificationService(lm_client=_lm_client())
    result = await service.classify_sets(
        text="my order is late",
        sets=[
            ClassificationSetDefinition("topic", _categories()),
            ClassificationSetDefinition(
                "channel",
                [
                    CategoryDefinition("complaint", "complaint"),
                    CategoryDefinition("question", "question"),
                ],
            ),
        ],
    )
    assert isinstance(result, ClassificationSetResult)
    assert "topic" in result.classifications
    assert "channel" in result.classifications
    assert result.classifications["topic"].category == "shipping"
    assert result.classifications["channel"].category == "complaint"
    assert result.classifications["topic"].strategy_used == "llm"


@patch("rfnry_rag.reasoning.modules.classification.strategies.b")
async def test_classify_sets_batch(mock_b):
    """Batch multi-set classification runs concurrently."""
    single_set_result = SimpleNamespace(
        classifications=[
            SimpleNamespace(
                set_name="topic",
                category="shipping",
                confidence=0.9,
                reasoning="about delivery",
                runner_up="billing",
                runner_up_confidence=0.1,
            ),
        ]
    )
    mock_b.ClassifyTextSets = AsyncMock(return_value=single_set_result)
    service = ClassificationService(lm_client=_lm_client())
    sets = [ClassificationSetDefinition("topic", _categories())]
    results = await service.classify_sets_batch(["text 1", "text 2"], sets)
    assert len(results) == 2
    assert all(isinstance(r, ClassificationSetResult) for r in results)


@patch("rfnry_rag.reasoning.modules.classification.strategies.b")
async def test_classify_sets_validates_category_names(mock_b):
    """Multi-set classification validates returned categories against set definitions."""
    bad_result = SimpleNamespace(
        classifications=[
            SimpleNamespace(
                set_name="topic",
                category="INVALID",
                confidence=0.9,
                reasoning="test",
                runner_up=None,
                runner_up_confidence=None,
            ),
        ]
    )
    mock_b.ClassifyTextSets = AsyncMock(return_value=bad_result)
    service = ClassificationService(lm_client=_lm_client())
    with pytest.raises(ClassificationError, match="invalid category"):
        await service.classify_sets(
            text="test",
            sets=[ClassificationSetDefinition("topic", _categories())],
        )


@patch("rfnry_rag.reasoning.modules.classification.strategies.b")
async def test_confidence_flagging(mock_b):
    """Low confidence results get needs_review=True."""
    mock_b.ClassifyText = AsyncMock(return_value=_mock_classify_result(confidence=0.3))
    service = ClassificationService(lm_client=_lm_client())
    result = await service.classify(
        "ambiguous text",
        _categories(),
        config=ClassificationConfig(low_confidence_threshold=0.5),
    )
    assert result.needs_review is True


@patch("rfnry_rag.reasoning.modules.classification.strategies.b")
async def test_confidence_above_threshold(mock_b):
    """High confidence results get needs_review=False."""
    mock_b.ClassifyText = AsyncMock(return_value=_mock_classify_result(confidence=0.9))
    service = ClassificationService(lm_client=_lm_client())
    result = await service.classify(
        "clear text",
        _categories(),
        config=ClassificationConfig(low_confidence_threshold=0.5),
    )
    assert result.needs_review is False


@patch("rfnry_rag.reasoning.modules.classification.strategies.b")
async def test_no_threshold_no_flag(mock_b):
    """Without threshold config, needs_review stays False."""
    mock_b.ClassifyText = AsyncMock(return_value=_mock_classify_result(confidence=0.3))
    service = ClassificationService(lm_client=_lm_client())
    result = await service.classify("test", _categories())
    assert result.needs_review is False


def test_classification_config_low_confidence_validation():
    """low_confidence_threshold must be 0.0-1.0."""
    with pytest.raises(ValueError, match="low_confidence_threshold"):
        ClassificationConfig(low_confidence_threshold=1.5)
