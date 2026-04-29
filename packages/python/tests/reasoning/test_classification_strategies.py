from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from rfnry_rag.reasoning.common.errors import ClassificationError
from rfnry_rag.reasoning.common.language_model import LanguageModelClient, LanguageModelProvider
from rfnry_rag.reasoning.modules.classification.models import (
    CategoryDefinition,
    Classification,
    ClassificationConfig,
    ClassificationSetDefinition,
)
from rfnry_rag.reasoning.modules.classification.service import ClassificationService
from rfnry_rag.reasoning.modules.classification.strategies import format_categories, format_category_sets


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


class _MockEmbeddings:
    model = "test-model"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(i)] * 8 for i in range(len(texts))]

    async def embedding_dimension(self) -> int:
        return 8


class _MockVectorStore:
    def __init__(self, results: list[dict[str, Any]] | None = None):
        self._results = (
            [
                {"category": "shipping", "text": "tracking info"},
                {"category": "shipping", "text": "delivery update"},
                {"category": "billing", "text": "invoice question"},
            ]
            if results is None
            else results
        )

    async def scroll(self, **_: Any) -> tuple[list, str | None]:
        return [], None

    async def search(self, **_: Any) -> list[dict]:
        return self._results


@patch("rfnry_rag.reasoning.modules.classification.strategies.b")
async def test_llm_classify_single(mock_b):
    mock_b.ClassifyText = AsyncMock(return_value=_mock_classify_result())
    service = ClassificationService(lm_client=_lm_client())
    result = await service.classify("where is my order", _categories())
    assert result.category == "shipping"
    assert result.confidence == pytest.approx(0.9)
    assert result.strategy_used == "llm"
    assert result.runner_up == "billing"


@patch("rfnry_rag.reasoning.modules.classification.strategies.b")
async def test_llm_classify_case_insensitive_fallback(mock_b):
    mock_b.ClassifyText = AsyncMock(return_value=_mock_classify_result(category="SHIPPING"))
    service = ClassificationService(lm_client=_lm_client())
    result = await service.classify("test", _categories())
    assert result.category == "shipping"


@patch("rfnry_rag.reasoning.modules.classification.strategies.b")
async def test_llm_classify_invalid_category_raises(mock_b):
    mock_b.ClassifyText = AsyncMock(return_value=_mock_classify_result(category="NONEXISTENT"))
    service = ClassificationService(lm_client=_lm_client())
    with pytest.raises(ClassificationError, match="invalid category"):
        await service.classify("test", _categories())


@patch("rfnry_rag.reasoning.modules.classification.strategies.b")
async def test_llm_classify_wraps_exceptions(mock_b):
    mock_b.ClassifyText = AsyncMock(side_effect=RuntimeError("api down"))
    service = ClassificationService(lm_client=_lm_client())
    with pytest.raises(ClassificationError, match="Classification failed"):
        await service.classify("test", _categories())


async def test_classify_without_lm_client_raises():
    service = ClassificationService()
    with pytest.raises(ClassificationError, match="requires lm_client"):
        await service.classify("test", _categories())


@patch("rfnry_rag.reasoning.modules.classification.strategies.b")
async def test_classify_batch_llm(mock_b):
    mock_b.ClassifyText = AsyncMock(return_value=_mock_classify_result())
    service = ClassificationService(lm_client=_lm_client())
    results = await service.classify_batch(["text 1", "text 2", "text 3"], _categories())
    assert len(results) == 3
    assert all(r.strategy_used == "llm" for r in results)


@patch("rfnry_rag.reasoning.modules.classification.strategies.b")
async def test_classify_batch_with_metadata(mock_b):
    mock_b.ClassifyText = AsyncMock(return_value=_mock_classify_result())
    service = ClassificationService(lm_client=_lm_client())
    metadata = [{"source": "email"}, {"source": "chat"}]
    results = await service.classify_batch(["t1", "t2"], _categories(), metadata=metadata)
    assert results[0].metadata == {"source": "email"}
    assert results[1].metadata == {"source": "chat"}


async def test_classify_batch_metadata_length_mismatch():
    service = ClassificationService(lm_client=_lm_client())
    with pytest.raises(ClassificationError, match="metadata length"):
        await service.classify_batch(["t1", "t2"], _categories(), metadata=[{"a": 1}])


async def test_knn_classify():
    store = _MockVectorStore()
    service = ClassificationService(
        embeddings=_MockEmbeddings(),
        lm_client=_lm_client(),
        vector_store=store,
    )
    result = await service.classify(
        "delivery question",
        _categories(),
        config=ClassificationConfig(strategy="hybrid", knn_knowledge_id="kb", escalation_threshold=0.0),
    )
    assert result.category == "shipping"
    assert result.vote_distribution is not None
    assert result.vote_distribution["shipping"] == 2


async def test_knn_no_results_raises():
    store = _MockVectorStore(results=[])
    service = ClassificationService(
        embeddings=_MockEmbeddings(),
        lm_client=_lm_client(),
        vector_store=store,
    )
    with pytest.raises(ClassificationError, match="No kNN results"):
        await service.classify(
            "test",
            _categories(),
            config=ClassificationConfig(strategy="hybrid", knn_knowledge_id="kb", escalation_threshold=0.0),
        )


@patch("rfnry_rag.reasoning.modules.classification.strategies.b")
async def test_hybrid_knn_confident_no_escalation(mock_b):
    store = _MockVectorStore(results=[{"category": "shipping"}] * 8 + [{"category": "billing"}] * 2)
    service = ClassificationService(
        embeddings=_MockEmbeddings(),
        lm_client=_lm_client(),
        vector_store=store,
    )
    result = await service.classify(
        "test",
        _categories(),
        config=ClassificationConfig(strategy="hybrid", knn_knowledge_id="kb", escalation_threshold=0.5),
    )
    assert result.strategy_used == "hybrid_knn"
    mock_b.ClassifyText.assert_not_called()


@patch("rfnry_rag.reasoning.modules.classification.strategies.b")
async def test_hybrid_escalates_to_llm(mock_b):
    mock_b.ClassifyText = AsyncMock(return_value=_mock_classify_result(category="product", confidence=0.95))
    store = _MockVectorStore(
        results=[
            {"category": "shipping"},
            {"category": "billing"},
            {"category": "product"},
        ]
    )
    service = ClassificationService(
        embeddings=_MockEmbeddings(),
        lm_client=_lm_client(),
        vector_store=store,
    )
    result = await service.classify(
        "test",
        _categories(),
        config=ClassificationConfig(strategy="hybrid", knn_knowledge_id="kb", escalation_threshold=0.9),
    )
    assert result.strategy_used == "hybrid_llm_escalation"
    assert result.category == "product"


async def test_hybrid_requires_embeddings():
    service = ClassificationService(lm_client=_lm_client())
    with pytest.raises(ClassificationError, match="requires embeddings"):
        await service.classify(
            "test",
            _categories(),
            config=ClassificationConfig(strategy="hybrid", knn_knowledge_id="kb"),
        )


async def test_hybrid_requires_vector_store():
    service = ClassificationService(embeddings=_MockEmbeddings(), lm_client=_lm_client())
    with pytest.raises(ClassificationError, match="requires vector_store"):
        await service.classify(
            "test",
            _categories(),
            config=ClassificationConfig(strategy="hybrid", knn_knowledge_id="kb"),
        )


async def test_hybrid_requires_knowledge_id():
    service = ClassificationService(
        embeddings=_MockEmbeddings(),
        lm_client=_lm_client(),
        vector_store=_MockVectorStore(),
    )
    with pytest.raises(ClassificationError, match="requires knn_knowledge_id"):
        await service.classify("test", _categories(), config=ClassificationConfig(strategy="hybrid"))


@patch("rfnry_rag.reasoning.modules.classification.strategies.b")
async def test_classify_sets_flags_low_confidence(mock_b):
    mock_b.ClassifyTextSets = AsyncMock(
        return_value=SimpleNamespace(
            classifications=[
                SimpleNamespace(
                    set_name="routing",
                    category="billing",
                    confidence=0.3,
                    reasoning="unsure",
                    runner_up=None,
                    runner_up_confidence=None,
                ),
            ]
        )
    )
    service = ClassificationService(lm_client=_lm_client())
    result = await service.classify_sets(
        "test",
        sets=[ClassificationSetDefinition("routing", _categories())],
        config=ClassificationConfig(low_confidence_threshold=0.5),
    )
    assert result.classifications["routing"].needs_review is True


def test_format_categories_basic():
    cats = [CategoryDefinition("a", "desc a"), CategoryDefinition("b", "desc b")]
    formatted = format_categories(cats)
    assert "**a**" in formatted
    assert "desc a" in formatted
    assert "**b**" in formatted


def test_format_categories_with_examples():
    cats = [CategoryDefinition("a", "desc", examples=["ex1", "ex2"])]
    formatted = format_categories(cats)
    assert '"ex1"' in formatted
    assert '"ex2"' in formatted


def test_format_category_sets():
    sets = [
        ClassificationSetDefinition("topic", [CategoryDefinition("a", "desc")]),
        ClassificationSetDefinition("channel", [CategoryDefinition("b", "desc")]),
    ]
    formatted = format_category_sets(sets)
    assert "=== Set: topic ===" in formatted
    assert "=== Set: channel ===" in formatted


def test_classification_to_dict_minimal():
    c = Classification(category="shipping", confidence=0.9, strategy_used="llm")
    d = c.to_dict()
    assert d == {"category": "shipping", "confidence": 0.9, "strategy_used": "llm"}
    assert "reasoning" not in d
    assert "runner_up" not in d


def test_classification_to_dict_full():
    c = Classification(
        category="billing",
        confidence=0.8,
        strategy_used="knn",
        reasoning="invoices",
        runner_up="shipping",
        runner_up_confidence=0.15,
        vote_distribution={"billing": 7, "shipping": 3},
        evidence=[{"text": "invoice"}],
    )
    d = c.to_dict()
    assert d["reasoning"] == "invoices"
    assert d["runner_up"] == "shipping"
    assert d["vote_distribution"]["billing"] == 7


def test_config_invalid_strategy():
    with pytest.raises(ValueError, match="Unknown strategy"):
        ClassificationConfig(strategy="invalid")


def test_config_escalation_threshold_out_of_range():
    with pytest.raises(ValueError, match="escalation_threshold"):
        ClassificationConfig(escalation_threshold=1.5)


def test_config_top_k_below_1():
    with pytest.raises(ValueError, match="top_k"):
        ClassificationConfig(top_k=0)
