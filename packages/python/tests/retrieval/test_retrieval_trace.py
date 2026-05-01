"""RetrievalTrace dataclass + opt-in trace=True flag.

These tests exercise the trace-collection plumbing at both the
`RetrievalService.retrieve()` level and the `RagEngine.query()` level.
They specifically verify the `None` vs `[]` discipline that gates the
failure-classification SCOPE_MISS / DRIFT verdicts — conflating "stage
did not run" with "stage ran and produced no results" would erase the
signal.
"""

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from rfnry_rag.config import RagEngineConfig, RoutingConfig
from rfnry_rag.generation.models import QueryResult
from rfnry_rag.models import RetrievedChunk
from rfnry_rag.retrieval.search.service import RetrievalService
from rfnry_rag.server import RagEngine


def _chunk(chunk_id: str, score: float = 0.9) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=chunk_id, source_id="s1", content="text", score=score)


def _mock_method(name: str, results: list[RetrievedChunk]) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        weight=1.0,
        top_k=None,
        search=AsyncMock(return_value=results),
    )


def _query_result(answer: str = "an answer") -> QueryResult:
    return QueryResult(answer=answer, sources=[], grounded=True, confidence=0.85)


def _make_engine_for_query(retrieve_return: tuple[list[RetrievedChunk], Any]) -> RagEngine:
    """Build a minimally-wired RagEngine for query() tests.

    Bypasses initialize() the same way test_server_query does. The
    `_retrieval_service.retrieve` mock returns the supplied tuple verbatim,
    so callers control whether trace is None or a populated RetrievalTrace.
    """
    config = MagicMock(spec=RagEngineConfig)
    config.retrieval = SimpleNamespace(history_window=3)
    config.routing = RoutingConfig()
    from rfnry_rag.observability import NullSink as _ObsNullSink
    from rfnry_rag.observability import Observability
    from rfnry_rag.telemetry import NullSink as _TelNullSink
    from rfnry_rag.telemetry import Telemetry

    server = RagEngine.__new__(RagEngine)
    server._config = config
    server._observability = Observability(sink=_ObsNullSink())
    server._telemetry = Telemetry(sink=_TelNullSink())
    server._initialized = True
    server._retrieval_service = AsyncMock()
    server._retrieval_service.retrieve = AsyncMock(return_value=retrieve_return)
    server._structured_retrieval = None
    server._generation_service = AsyncMock()
    server._generation_service.generate = AsyncMock(return_value=_query_result())
    server._knowledge_manager = None
    server._ingestion_service = None
    server._structured_ingestion = None
    server._retrieval_namespace = None
    server._ingestion_namespace = None
    return server


# --- Engine-level: opt-out / opt-in semantics ---


async def test_retrieve_default_returns_no_trace() -> None:
    """`engine.query(...)` without trace=True returns QueryResult with trace=None."""
    server = _make_engine_for_query(([_chunk("c1")], None))
    result = await server.query("hello")
    assert result.trace is None


async def test_retrieve_with_trace_populates_query_field() -> None:
    """`engine.query(text, trace=True)` populates trace.query with the original text."""
    from rfnry_rag.observability.trace import RetrievalTrace

    incoming_trace = RetrievalTrace(query="hello")
    server = _make_engine_for_query(([_chunk("c1")], incoming_trace))
    result = await server.query("hello", trace=True)
    assert result.trace is not None
    assert result.trace.query == "hello"


# --- Service-level: per-stage population ---


async def test_retrieve_trace_per_method_dict_keys_match_method_names() -> None:
    """Trace's per_method_results is keyed by `BaseRetrievalMethod.name`."""
    method_a = _mock_method("method_a", [_chunk("chunk_a")])
    method_b = _mock_method("method_b", [_chunk("chunk_b")])
    service = RetrievalService(retrieval_methods=[method_a, method_b], top_k=5)

    _chunks, trace = await service.retrieve(query="q", trace=True)

    assert trace is not None
    assert set(trace.per_method_results.keys()) == {"method_a", "method_b"}


async def test_retrieve_trace_per_method_includes_empty_results_method() -> None:
    """A method that ran and returned [] still appears in the dict, with [] value."""
    method_a = _mock_method("method_a", [_chunk("chunk_a")])
    method_b = _mock_method("method_b", [])  # ran, returned nothing
    service = RetrievalService(retrieval_methods=[method_a, method_b], top_k=5)

    _chunks, trace = await service.retrieve(query="q", trace=True)

    assert trace is not None
    assert "method_a" in trace.per_method_results
    assert "method_b" in trace.per_method_results
    assert trace.per_method_results["method_b"] == []
    assert trace.per_method_results["method_a"] != []


async def test_retrieve_trace_reranked_none_when_disabled() -> None:
    """No reranker => reranked_results is None.

    This is the load-bearing distinction: None means "stage did not run",
    not "stage ran with no results".
    """
    method_a = _mock_method("method_a", [_chunk("chunk_a")])
    service = RetrievalService(retrieval_methods=[method_a], top_k=5)

    _chunks, trace = await service.retrieve(query="q", trace=True)

    assert trace is not None
    assert trace.reranked_results is None


async def test_retrieve_trace_timings_only_includes_stages_that_ran() -> None:
    """No rewriter, no reranker => timings only has retrieval + fusion."""
    method_a = _mock_method("method_a", [_chunk("chunk_a")])
    service = RetrievalService(retrieval_methods=[method_a], top_k=5)

    _chunks, trace = await service.retrieve(query="q", trace=True)

    assert trace is not None
    assert set(trace.timings.keys()) == {"retrieval", "fusion"}


async def test_retrieve_trace_routing_decision_is_none_placeholder() -> None:
    """Default value before any stage populates it; routing modes populate it during query dispatch."""
    method_a = _mock_method("method_a", [_chunk("chunk_a")])
    service = RetrievalService(retrieval_methods=[method_a], top_k=5)

    _chunks, trace = await service.retrieve(query="q", trace=True)

    assert trace is not None
    assert trace.routing_decision is None


async def test_rag_engine_retrieve_with_trace_returns_trace_alongside_chunks() -> None:
    """`engine.retrieve(text, trace=True)` returns (chunks, trace) with the
    raw-retrieval-trace shape: grounding_decision and confidence are None
    (no grounding stage runs), and final_results carries the chunks the
    caller actually receives.
    """
    from rfnry_rag.observability.trace import RetrievalTrace

    incoming_chunks = [_chunk("c1")]
    incoming_trace = RetrievalTrace(query="query", final_results=list(incoming_chunks))
    server = _make_engine_for_query((incoming_chunks, incoming_trace))

    chunks, trace = await server.retrieve("query", trace=True)

    assert chunks == incoming_chunks
    assert trace is not None
    assert trace.query == "query"
    assert trace.grounding_decision is None
    assert trace.confidence is None
    assert trace.final_results  # populated with the chunks the caller receives
    assert trace.final_results[0].chunk_id == "c1"
