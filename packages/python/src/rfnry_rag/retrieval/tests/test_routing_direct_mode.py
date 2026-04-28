"""R1.2 — QueryMode + RoutingConfig + DIRECT context mode dispatch.

Lights up `mode="direct"` user-facing. RETRIEVAL stays the default (backward
compat); HYBRID and AUTO raise `ConfigurationError` until R1.3 / R1.4 land.
DIRECT skips retrieval, loads the full corpus via R1.1's `_load_full_corpus`,
and routes through `GenerationService.generate_from_corpus` (gates skipped —
the entire corpus is in the prompt).
"""

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from rfnry_rag.retrieval.common.errors import ConfigurationError
from rfnry_rag.retrieval.common.models import RetrievalTrace
from rfnry_rag.retrieval.modules.generation.models import QueryResult
from rfnry_rag.retrieval.server import (
    QueryMode,
    RagEngine,
    RagServerConfig,
    RoutingConfig,
)


def _query_result(answer: str = "an answer") -> QueryResult:
    return QueryResult(answer=answer, sources=[], grounded=True, confidence=0.85)


def _make_engine(*, mode: QueryMode) -> Any:
    """Build a minimally-wired RagEngine bypassing initialize().

    Returns `Any` rather than `RagEngine` so tests can poke `AsyncMock`
    assertion helpers (`assert_awaited_once`, etc.) on private service
    attributes typed as concrete services in the engine class.
    """
    config = MagicMock(spec=RagServerConfig)
    config.retrieval = SimpleNamespace(history_window=3)
    config.routing = RoutingConfig(mode=mode)

    engine = RagEngine.__new__(RagEngine)
    engine._config = config
    engine._initialized = True
    engine._retrieval_service = AsyncMock()
    cast(Any, engine._retrieval_service).retrieve = AsyncMock(return_value=([], None))
    engine._structured_retrieval = None
    engine._generation_service = AsyncMock()
    cast(Any, engine._generation_service).generate = AsyncMock(return_value=_query_result())
    cast(Any, engine._generation_service).generate_from_corpus = AsyncMock(return_value=_query_result())
    engine._step_service = None
    engine._knowledge_manager = None
    engine._ingestion_service = None
    engine._structured_ingestion = None
    engine._retrieval_namespace = None
    engine._ingestion_namespace = None
    engine._tree_indexing_service = None
    engine._tree_search_service = None
    return engine


def test_routing_config_default_mode_is_retrieval() -> None:
    """`RoutingConfig()` default mode is RETRIEVAL (backward compat)."""
    config = RoutingConfig()
    assert config.mode == QueryMode.RETRIEVAL


def test_routing_config_direct_context_threshold_bounded() -> None:
    """`direct_context_threshold` bounded `1_000 ≤ n ≤ 2_000_000`."""
    with pytest.raises(ConfigurationError):
        RoutingConfig(direct_context_threshold=999)
    with pytest.raises(ConfigurationError):
        RoutingConfig(direct_context_threshold=2_000_001)
    # Boundary values + nominal default succeed.
    RoutingConfig(direct_context_threshold=1_000)
    RoutingConfig(direct_context_threshold=150_000)
    RoutingConfig(direct_context_threshold=2_000_000)


async def test_query_mode_retrieval_uses_existing_pipeline() -> None:
    """`mode=RETRIEVAL` goes through `_retrieve_chunks` + generation."""
    engine = _make_engine(mode=QueryMode.RETRIEVAL)
    engine._load_full_corpus = AsyncMock(return_value="should not be called")  # type: ignore[method-assign]

    await engine.query("q1", knowledge_id="kb-1")

    engine._retrieval_service.retrieve.assert_awaited_once()
    engine._load_full_corpus.assert_not_called()
    engine._generation_service.generate.assert_awaited_once()
    engine._generation_service.generate_from_corpus.assert_not_called()


async def test_query_mode_direct_skips_retrieval_and_loads_full_corpus() -> None:
    """`mode=DIRECT` skips retrieval and calls `_load_full_corpus(knowledge_id)`."""
    engine = _make_engine(mode=QueryMode.DIRECT)
    engine._load_full_corpus = AsyncMock(return_value="full corpus body")  # type: ignore[method-assign]

    await engine.query("q1", knowledge_id="kb-1")

    engine._retrieval_service.retrieve.assert_not_called()
    engine._load_full_corpus.assert_awaited_once_with("kb-1")
    engine._generation_service.generate_from_corpus.assert_awaited_once()
    engine._generation_service.generate.assert_not_called()


async def test_query_mode_direct_sets_routing_decision_in_trace() -> None:
    """DIRECT populates `trace.routing_decision = "direct"`; RETRIEVAL sets `"retrieval"`."""
    direct = _make_engine(mode=QueryMode.DIRECT)
    direct._load_full_corpus = AsyncMock(return_value="corpus")  # type: ignore[method-assign]
    direct_result = await direct.query("q1", knowledge_id="kb-1", trace=True)
    assert direct_result.trace is not None
    assert direct_result.trace.routing_decision == "direct"

    retrieval = _make_engine(mode=QueryMode.RETRIEVAL)
    retrieval._retrieval_service.retrieve = AsyncMock(
        return_value=([], RetrievalTrace(query="q1", knowledge_id="kb-1"))
    )
    retrieval_result = await retrieval.query("q1", knowledge_id="kb-1", trace=True)
    assert retrieval_result.trace is not None
    assert retrieval_result.trace.routing_decision == "retrieval"


async def test_query_mode_auto_raises_not_implemented_in_r1_3() -> None:
    """AUTO raises ConfigurationError pointing to R1.4 (HYBRID is live in R1.3)."""
    auto = _make_engine(mode=QueryMode.AUTO)
    with pytest.raises(ConfigurationError, match="not yet implemented"):
        await auto.query("q1", knowledge_id="kb-1")


async def test_query_mode_direct_returns_query_result_shape_unchanged() -> None:
    """DIRECT returns a `QueryResult` with `answer` populated and trace shape honest."""
    engine = _make_engine(mode=QueryMode.DIRECT)
    engine._load_full_corpus = AsyncMock(return_value="corpus")  # type: ignore[method-assign]

    result = await engine.query("q1", knowledge_id="kb-1", trace=True)

    assert isinstance(result, QueryResult)
    assert result.answer
    assert result.trace is not None
    # DIRECT doesn't retrieve; final_results is honest about that.
    assert result.trace.final_results == []
    # DIRECT has no chunk-level attribution — `sources` must be empty.
    assert result.sources == []
