from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from rfnry_rag.reasoning.common.errors import AnalysisError
from rfnry_rag.reasoning.common.language_model import LanguageModelClient, LanguageModelProvider
from rfnry_rag.reasoning.modules.analysis.models import (
    AnalysisConfig,
    AnalysisResult,
    ContextTrackingConfig,
    DimensionDefinition,
    EntityTypeDefinition,
    Message,
)
from rfnry_rag.reasoning.modules.analysis.service import AnalysisService


def _lm_client() -> LanguageModelClient:
    return LanguageModelClient(
        provider=LanguageModelProvider(provider="openai", model="gpt-4o-mini", api_key="test-key"),
    )


def _mock_text_result(
    *,
    dimensions: list | None = None,
    entities: list | None = None,
    hints: list | None = None,
    summary: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        primary_intent="delivery inquiry",
        confidence=0.92,
        summary=summary,
        dimensions=dimensions or [],
        entities=entities or [],
        retrieval_hints=hints or [],
    )


def _mock_thread_result(
    *,
    dimensions: list | None = None,
    entities: list | None = None,
    hints: list | None = None,
    summary: str | None = None,
    intent_shifts: list | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        primary_intent="order complaint",
        confidence=0.88,
        summary=summary,
        dimensions=dimensions or [],
        entities=entities or [],
        retrieval_hints=hints or [],
        intent_shifts=intent_shifts or [],
        escalation_detected=True,
        escalation_reasoning="customer expressed frustration",
        resolution_status="pending",
    )


@patch("rfnry_rag.reasoning.modules.analysis.service.b")
async def test_analyze_minimal(mock_b):
    """Analyze with no config returns intent and confidence only."""
    mock_b.AnalyzeText = AsyncMock(return_value=_mock_text_result())
    service = AnalysisService(lm_client=_lm_client())
    result = await service.analyze("test text")
    assert isinstance(result, AnalysisResult)
    assert result.primary_intent == "delivery inquiry"
    assert result.confidence == pytest.approx(0.92)
    assert result.dimensions == {}
    assert result.entities == []
    assert result.summary is None
    assert result.retrieval_hints == []


@patch("rfnry_rag.reasoning.modules.analysis.service.b")
async def test_analyze_with_dimensions(mock_b):
    """Analyze with consumer-defined dimensions returns scored results."""
    mock_b.AnalyzeText = AsyncMock(
        return_value=_mock_text_result(
            dimensions=[
                SimpleNamespace(name="urgency", value="0.85", confidence=0.9, reasoning="time pressure"),
                SimpleNamespace(name="sentiment", value="negative", confidence=0.8, reasoning="frustration"),
            ]
        )
    )
    service = AnalysisService(lm_client=_lm_client())
    result = await service.analyze(
        "my order is late",
        config=AnalysisConfig(
            dimensions=[
                DimensionDefinition("urgency", "How time-sensitive", "0.0-1.0"),
                DimensionDefinition("sentiment", "Emotional tone", "negative/neutral/positive"),
            ]
        ),
    )
    assert "urgency" in result.dimensions
    assert result.dimensions["urgency"].value == "0.85"
    assert result.dimensions["urgency"].reasoning == "time pressure"
    assert "sentiment" in result.dimensions
    assert result.dimensions["sentiment"].value == "negative"


@patch("rfnry_rag.reasoning.modules.analysis.service.b")
async def test_analyze_with_entities(mock_b):
    """Analyze with entity types returns extracted entities."""
    mock_b.AnalyzeText = AsyncMock(
        return_value=_mock_text_result(
            entities=[
                SimpleNamespace(type="order_id", value="ORD-123", context="my order ORD-123"),
            ]
        )
    )
    service = AnalysisService(lm_client=_lm_client())
    result = await service.analyze(
        "my order ORD-123 is late",
        config=AnalysisConfig(entity_types=[EntityTypeDefinition("order_id", "Order identifier")]),
    )
    assert len(result.entities) == 1
    assert result.entities[0].type == "order_id"
    assert result.entities[0].value == "ORD-123"


@patch("rfnry_rag.reasoning.modules.analysis.service.b")
async def test_analyze_with_summary(mock_b):
    """Analyze with summarize=True returns summary."""
    mock_b.AnalyzeText = AsyncMock(return_value=_mock_text_result(summary="Customer reports delayed delivery"))
    service = AnalysisService(lm_client=_lm_client())
    result = await service.analyze("my order is late", config=AnalysisConfig(summarize=True))
    assert result.summary == "Customer reports delayed delivery"


@patch("rfnry_rag.reasoning.modules.analysis.service.b")
async def test_analyze_with_retrieval_hints(mock_b):
    """Analyze with retrieval hints returns scoped suggestions."""
    mock_b.AnalyzeText = AsyncMock(
        return_value=_mock_text_result(
            hints=[
                SimpleNamespace(
                    query="shipping delay policy",
                    knowledge_scope="policies",
                    reasoning="delayed order",
                    priority=0.9,
                ),
            ]
        )
    )
    service = AnalysisService(lm_client=_lm_client())
    result = await service.analyze(
        "my order is late",
        config=AnalysisConfig(
            generate_retrieval_hints=True,
            retrieval_hint_scopes=["policies"],
        ),
    )
    assert len(result.retrieval_hints) == 1
    assert result.retrieval_hints[0].knowledge_scope == "policies"
    assert result.retrieval_hints[0].priority == pytest.approx(0.9)


@patch("rfnry_rag.reasoning.modules.analysis.service.b")
async def test_analyze_context_with_tracking(mock_b):
    """Analyze thread with tracking returns intent shifts and escalation."""
    mock_b.AnalyzeContext = AsyncMock(
        return_value=_mock_thread_result(
            intent_shifts=[
                SimpleNamespace(
                    from_intent="inquiry",
                    to_intent="complaint",
                    at_message=2,
                    reasoning="tone changed",
                ),
            ]
        )
    )
    service = AnalysisService(lm_client=_lm_client())
    messages = [
        Message(text="where is my order?", role="customer"),
        Message(text="let me check", role="agent"),
        Message(text="this is unacceptable!", role="customer"),
    ]
    result = await service.analyze_context(
        messages,
        config=AnalysisConfig(context_tracking=ContextTrackingConfig()),
    )
    assert result.primary_intent == "order complaint"
    assert result.escalation_detected is True
    assert result.resolution_status == "pending"
    assert len(result.intent_shifts) == 1
    assert result.intent_shifts[0].from_intent == "inquiry"
    assert result.intent_shifts[0].at_message == 2


@patch("rfnry_rag.reasoning.modules.analysis.service.b")
async def test_analyze_context_without_tracking(mock_b):
    """Analyze thread without tracking still returns base analysis."""
    mock_b.AnalyzeContext = AsyncMock(return_value=_mock_thread_result())
    service = AnalysisService(lm_client=_lm_client())
    messages = [Message(text="hello", role="customer")]
    result = await service.analyze_context(messages)
    assert result.primary_intent == "order complaint"
    assert result.escalation_detected is True


@patch("rfnry_rag.reasoning.modules.analysis.service.b")
async def test_analyze_batch(mock_b):
    """Batch analysis runs concurrently and returns ordered results."""
    mock_b.AnalyzeText = AsyncMock(return_value=_mock_text_result())
    service = AnalysisService(lm_client=_lm_client())
    results = await service.analyze_batch(["text 1", "text 2", "text 3"])
    assert len(results) == 3
    assert all(isinstance(r, AnalysisResult) for r in results)


@patch("rfnry_rag.reasoning.modules.analysis.service.b")
async def test_analyze_truncates_text(mock_b):
    """Text longer than max_text_length is truncated."""
    mock_b.AnalyzeText = AsyncMock(return_value=_mock_text_result())
    service = AnalysisService(lm_client=_lm_client())
    long_text = "x" * 5000
    await service.analyze(long_text, config=AnalysisConfig(max_text_length=100))
    call_args = mock_b.AnalyzeText.call_args
    assert len(call_args[0][0]) == 100


@patch("rfnry_rag.reasoning.modules.analysis.service.b")
async def test_analyze_raises_analysis_error(mock_b):
    """Service wraps unexpected exceptions in AnalysisError."""
    mock_b.AnalyzeText = AsyncMock(side_effect=RuntimeError("boom"))
    service = AnalysisService(lm_client=_lm_client())
    with pytest.raises(AnalysisError, match="Analysis failed"):
        await service.analyze("test")


def test_analysis_config_validation():
    """Config validates retrieval hints require scopes."""
    with pytest.raises(ValueError, match="retrieval_hint_scopes required"):
        AnalysisConfig(generate_retrieval_hints=True)

    with pytest.raises(ValueError, match="max_text_length"):
        AnalysisConfig(max_text_length=0)


def test_analysis_config_max_text_length_upper_bound() -> None:
    with pytest.raises(ValueError, match="max_text_length must be <= "):
        AnalysisConfig(max_text_length=5_000_001)
