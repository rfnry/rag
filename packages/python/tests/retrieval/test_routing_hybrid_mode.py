"""HYBRID mode (SELF-ROUTE) tests.

`mode="hybrid"` runs RAG first, then asks the LLM "can you answer from
these chunks?" via `b.CheckAnswerability`. If yes, returns the RAG answer
(`routing_decision="hybrid_rag"`). If no, escalates to DIRECT-style
full-corpus generation (`routing_decision="hybrid_lc"`). On answerability-
check failure, degrades to RAG to avoid silent long-context escalation on
a transient error (rate limit, timeout, malformed JSON).
"""

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import pytest

from rfnry_rag.common.language_model import LanguageModelClient, LanguageModelProvider
from rfnry_rag.retrieval.common.errors import ConfigurationError
from rfnry_rag.retrieval.common.formatting import ChunkOrdering
from rfnry_rag.retrieval.common.models import RetrievalTrace, RetrievedChunk
from rfnry_rag.retrieval.modules.generation.models import QueryResult
from rfnry_rag.retrieval.server import (
    QueryMode,
    RoutingConfig,
)


def _stub_lm_client() -> LanguageModelClient:
    """Minimal LanguageModelClient instance — fields validate but no calls run."""
    return LanguageModelClient(
        provider=LanguageModelProvider(
            provider="anthropic",
            model="claude-sonnet-4-5",
            api_key="sk-test",
        )
    )


def _query_result(answer: str = "rag answer") -> QueryResult:
    return QueryResult(answer=answer, sources=[], grounded=True, confidence=0.85)


def _chunk(content: str = "chunk_a content", score: float = 0.9) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="chunk_a",
        source_id="src_a",
        content=content,
        score=score,
        metadata={},
    )


def _engine(
    make_engine: Any,
    *,
    mode: QueryMode,
    chunks: list[RetrievedChunk] | None = None,
    rag_answer: str = "rag answer",
    lc_answer: str = "lc answer",
) -> Any:
    """Build a HYBRID-mode engine via the shared ``make_engine`` factory.

    Wires per-test custom ``generate`` / ``generate_from_corpus`` answers,
    a chunk-returning ``_retrieve_chunks``, and the ``_answerability_registry``
    sentinel that the HYBRID arm reads.
    """
    routing_kwargs: dict[str, Any] = {"mode": mode}
    if mode == QueryMode.HYBRID:
        routing_kwargs["hybrid_answerability_model"] = _stub_lm_client()

    gs: Any = AsyncMock()
    cast(Any, gs).generate = AsyncMock(return_value=_query_result(rag_answer))
    cast(Any, gs).generate_from_corpus = AsyncMock(return_value=_query_result(lc_answer))

    engine = make_engine(
        retrieval=SimpleNamespace(history_window=3),
        generation=SimpleNamespace(chunk_ordering=ChunkOrdering.SCORE_DESCENDING),
        routing=RoutingConfig(**routing_kwargs),
        generation_service=gs,
        answerability_registry=object(),  # opaque sentinel — BAML call is mocked
    )
    chunks = chunks if chunks is not None else [_chunk()]
    engine._retrieve_chunks = AsyncMock(return_value=(chunks, None))  # type: ignore[method-assign]
    engine._load_full_corpus = AsyncMock(return_value="full corpus body")  # type: ignore[method-assign]
    return engine


def _verdict(answerable: bool, reasoning: str = "ok") -> SimpleNamespace:
    return SimpleNamespace(answerable=answerable, reasoning=reasoning)


def test_routing_config_hybrid_requires_answerability_model() -> None:
    """`mode=HYBRID` without `hybrid_answerability_model` raises; with stub succeeds."""
    with pytest.raises(ConfigurationError, match="hybrid_answerability_model"):
        RoutingConfig(mode=QueryMode.HYBRID, hybrid_answerability_model=None)
    # Providing a stub client succeeds.
    RoutingConfig(mode=QueryMode.HYBRID, hybrid_answerability_model=_stub_lm_client())


async def test_query_mode_hybrid_runs_rag_then_answerability_check(
    make_engine: Any,
) -> None:
    """HYBRID retrieves chunks AND calls `b.CheckAnswerability`."""
    engine = _engine(make_engine, mode=QueryMode.HYBRID)
    with patch(
        "rfnry_rag.retrieval.server.b.CheckAnswerability",
        new=AsyncMock(return_value=_verdict(True)),
    ) as mock_check:
        await engine.query("q1", knowledge_id="kb-1")

    engine._retrieve_chunks.assert_awaited_once()
    mock_check.assert_awaited_once()


async def test_query_mode_hybrid_returns_rag_answer_when_answerable(
    make_engine: Any,
) -> None:
    """answerable=True: RAG generation runs; full corpus NOT loaded."""
    engine = _engine(make_engine, mode=QueryMode.HYBRID, rag_answer="rag-path-answer")
    with patch(
        "rfnry_rag.retrieval.server.b.CheckAnswerability",
        new=AsyncMock(return_value=_verdict(True)),
    ):
        result = await engine.query("q1", knowledge_id="kb-1")

    engine._load_full_corpus.assert_not_called()
    engine._generation_service.generate.assert_awaited_once()
    engine._generation_service.generate_from_corpus.assert_not_called()
    assert result.answer == "rag-path-answer"


async def test_query_mode_hybrid_escalates_to_lc_when_not_answerable(
    make_engine: Any,
) -> None:
    """answerable=False: full corpus loaded, generate_from_corpus called, sources empty."""
    engine = _engine(make_engine, mode=QueryMode.HYBRID, lc_answer="lc-path-answer")
    with patch(
        "rfnry_rag.retrieval.server.b.CheckAnswerability",
        new=AsyncMock(return_value=_verdict(False, "missing")),
    ):
        result = await engine.query("q1", knowledge_id="kb-1")

    engine._load_full_corpus.assert_awaited_once_with("kb-1")
    engine._generation_service.generate_from_corpus.assert_awaited_once()
    engine._generation_service.generate.assert_not_called()
    assert result.answer == "lc-path-answer"
    assert result.sources == []


async def test_query_mode_hybrid_routing_decision_hybrid_rag_on_success(
    make_engine: Any,
) -> None:
    """answerable=True: trace.routing_decision == "hybrid_rag"."""
    engine = _engine(make_engine, mode=QueryMode.HYBRID)
    engine._retrieve_chunks = AsyncMock(  # type: ignore[method-assign]
        return_value=([_chunk()], RetrievalTrace(query="q1", knowledge_id="kb-1"))
    )
    with patch(
        "rfnry_rag.retrieval.server.b.CheckAnswerability",
        new=AsyncMock(return_value=_verdict(True)),
    ):
        result = await engine.query("q1", knowledge_id="kb-1", trace=True)

    assert result.trace is not None
    assert result.trace.routing_decision == "hybrid_rag"


async def test_query_mode_hybrid_routing_decision_hybrid_lc_on_escalation(
    make_engine: Any,
) -> None:
    """answerable=False: trace.routing_decision == "hybrid_lc"."""
    engine = _engine(make_engine, mode=QueryMode.HYBRID)
    engine._retrieve_chunks = AsyncMock(  # type: ignore[method-assign]
        return_value=([_chunk()], RetrievalTrace(query="q1", knowledge_id="kb-1"))
    )
    with patch(
        "rfnry_rag.retrieval.server.b.CheckAnswerability",
        new=AsyncMock(return_value=_verdict(False, "missing")),
    ):
        result = await engine.query("q1", knowledge_id="kb-1", trace=True)

    assert result.trace is not None
    assert result.trace.routing_decision == "hybrid_lc"


async def test_query_mode_hybrid_degrades_to_rag_on_check_exception(
    make_engine: Any,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """CheckAnswerability raising: degrade to RAG (no LC escalation), log warning."""
    engine = _engine(make_engine, mode=QueryMode.HYBRID)
    engine._retrieve_chunks = AsyncMock(  # type: ignore[method-assign]
        return_value=([_chunk()], RetrievalTrace(query="q1", knowledge_id="kb-1"))
    )
    with (
        patch(
            "rfnry_rag.retrieval.server.b.CheckAnswerability",
            new=AsyncMock(side_effect=RuntimeError("rate limit")),
        ),
        caplog.at_level("WARNING", logger="rfnry_rag.server"),
    ):
        result = await engine.query("q1", knowledge_id="kb-1", trace=True)

    engine._load_full_corpus.assert_not_called()
    engine._generation_service.generate.assert_awaited_once()
    assert result.trace is not None
    assert result.trace.routing_decision == "hybrid_rag"
    assert any("answerability check failed" in rec.message for rec in caplog.records)


async def test_query_mode_hybrid_trace_includes_answerability_timing(
    make_engine: Any,
) -> None:
    """trace.timings contains `answerability_check` with a positive float."""
    engine = _engine(make_engine, mode=QueryMode.HYBRID)
    engine._retrieve_chunks = AsyncMock(  # type: ignore[method-assign]
        return_value=([_chunk()], RetrievalTrace(query="q1", knowledge_id="kb-1"))
    )
    with patch(
        "rfnry_rag.retrieval.server.b.CheckAnswerability",
        new=AsyncMock(return_value=_verdict(True)),
    ):
        result = await engine.query("q1", knowledge_id="kb-1", trace=True)

    assert result.trace is not None
    assert "answerability_check" in result.trace.timings
    assert result.trace.timings["answerability_check"] >= 0.0


@pytest.mark.parametrize("mode", [QueryMode.DIRECT, QueryMode.HYBRID, QueryMode.AUTO])
async def test_query_stream_refuses_non_retrieval_modes(make_engine: Any, mode: QueryMode) -> None:
    """`query_stream()` raises `ConfigurationError` for any non-RETRIEVAL mode.

    A consumer who configures `mode=DIRECT` / `mode=HYBRID` / `mode=AUTO`
    and calls `query_stream(...)` would otherwise silently get RAG-only
    behavior. Streaming for non-retrieval modes is deferred — refuse
    explicitly.
    """
    engine = _engine(make_engine, mode=mode)
    with pytest.raises(ConfigurationError, match="does not support mode"):
        async for _event in engine.query_stream("q1", knowledge_id="kb-1"):
            pass
