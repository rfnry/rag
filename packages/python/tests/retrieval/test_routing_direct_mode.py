"""Tests for QueryMode + RoutingConfig + DIRECT context mode dispatch.

`mode="direct"` is user-facing; RETRIEVAL is the default. DIRECT skips
retrieval, loads the full corpus via `_load_full_corpus`, and routes through
`GenerationService.generate_from_corpus` (gates skipped — the entire corpus
is in the prompt).
"""

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from rfnry_rag.exceptions import ConfigurationError
from rfnry_rag.generation.models import QueryResult
from rfnry_rag.retrieval.common.models import RetrievalTrace
from rfnry_rag.server import (
    QueryMode,
    RoutingConfig,
)


def test_routing_config_default_mode_is_retrieval() -> None:
    """`RoutingConfig()` default mode is RETRIEVAL (backward compat)."""
    config = RoutingConfig()
    assert config.mode == QueryMode.RETRIEVAL


def test_routing_config_full_context_threshold_bounded() -> None:
    """`full_context_threshold` bounded `1_000 ≤ n ≤ 2_000_000`."""
    with pytest.raises(ConfigurationError):
        RoutingConfig(full_context_threshold=999)
    with pytest.raises(ConfigurationError):
        RoutingConfig(full_context_threshold=2_000_001)
    # Boundary values + nominal default succeed.
    RoutingConfig(full_context_threshold=1_000)
    RoutingConfig(full_context_threshold=150_000)
    RoutingConfig(full_context_threshold=2_000_000)


async def test_query_mode_retrieval_uses_existing_pipeline(make_engine: Any) -> None:
    """`mode=RETRIEVAL` goes through `_retrieve_chunks` + generation."""
    engine = make_engine(
        retrieval=SimpleNamespace(history_window=3),
        routing=RoutingConfig(mode=QueryMode.RETRIEVAL),
    )
    engine._load_full_corpus = AsyncMock(return_value="should not be called")  # type: ignore[method-assign]

    await engine.query("q1", knowledge_id="kb-1")

    engine._retrieval_service.retrieve.assert_awaited_once()
    engine._load_full_corpus.assert_not_called()
    engine._generation_service.generate.assert_awaited_once()
    engine._generation_service.generate_from_corpus.assert_not_called()


async def test_query_mode_direct_skips_retrieval_and_loads_full_corpus(
    make_engine: Any,
) -> None:
    """`mode=DIRECT` skips retrieval and calls `_load_full_corpus(knowledge_id)`."""
    engine = make_engine(
        retrieval=SimpleNamespace(history_window=3),
        routing=RoutingConfig(mode=QueryMode.DIRECT),
    )
    engine._load_full_corpus = AsyncMock(return_value="full corpus body")  # type: ignore[method-assign]

    await engine.query("q1", knowledge_id="kb-1")

    engine._retrieval_service.retrieve.assert_not_called()
    engine._load_full_corpus.assert_awaited_once_with("kb-1")
    engine._generation_service.generate_from_corpus.assert_awaited_once()
    engine._generation_service.generate.assert_not_called()


async def test_query_mode_direct_sets_routing_decision_in_trace(
    make_engine: Any,
) -> None:
    """DIRECT populates `trace.routing_decision = "direct"`; RETRIEVAL sets `"retrieval"`."""
    direct = make_engine(
        retrieval=SimpleNamespace(history_window=3),
        routing=RoutingConfig(mode=QueryMode.DIRECT),
    )
    direct._load_full_corpus = AsyncMock(return_value="corpus")  # type: ignore[method-assign]
    direct_result = await direct.query("q1", knowledge_id="kb-1", trace=True)
    assert direct_result.trace is not None
    assert direct_result.trace.routing_decision == "direct"

    retrieval = make_engine(
        retrieval=SimpleNamespace(history_window=3),
        routing=RoutingConfig(mode=QueryMode.RETRIEVAL),
    )
    retrieval._retrieval_service.retrieve = AsyncMock(
        return_value=([], RetrievalTrace(query="q1", knowledge_id="kb-1"))
    )
    retrieval_result = await retrieval.query("q1", knowledge_id="kb-1", trace=True)
    assert retrieval_result.trace is not None
    assert retrieval_result.trace.routing_decision == "retrieval"


async def test_query_mode_direct_returns_query_result_shape_unchanged(
    make_engine: Any,
) -> None:
    """DIRECT returns a `QueryResult` with `answer` populated and trace shape honest."""
    engine = make_engine(
        retrieval=SimpleNamespace(history_window=3),
        routing=RoutingConfig(mode=QueryMode.DIRECT),
    )
    engine._load_full_corpus = AsyncMock(return_value="corpus")  # type: ignore[method-assign]

    result = await engine.query("q1", knowledge_id="kb-1", trace=True)

    assert isinstance(result, QueryResult)
    assert result.answer
    assert result.trace is not None
    # DIRECT doesn't retrieve; final_results is honest about that.
    assert result.trace.final_results == []
    # DIRECT has no chunk-level attribution — `sources` must be empty.
    assert result.sources == []
