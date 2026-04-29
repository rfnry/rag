"""AUTO mode: token-count-based dispatch between RETRIEVAL and DIRECT.

`mode="auto"` is user-facing and recommended for new users. AUTO reads
`KnowledgeManager.get_corpus_tokens(knowledge_id)` and routes to DIRECT when
`tokens <= direct_context_threshold`, otherwise RETRIEVAL. AUTO never routes
to HYBRID by design — HYBRID adds an answerability LLM call that isn't
justified without benchmark data.
"""

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

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


def _make_engine(
    *,
    mode: QueryMode = QueryMode.AUTO,
    threshold: int = 150_000,
    corpus_tokens: int = 0,
) -> Any:
    """Build a minimally-wired RagEngine bypassing initialize().

    Returns `Any` so tests can poke `AsyncMock` assertion helpers on private
    service attributes typed as concrete services in the engine class.
    """
    config = MagicMock(spec=RagServerConfig)
    config.retrieval = SimpleNamespace(history_window=3)
    config.routing = RoutingConfig(mode=mode, direct_context_threshold=threshold)

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
    engine._knowledge_manager = MagicMock()
    cast(Any, engine._knowledge_manager).get_corpus_tokens = AsyncMock(return_value=corpus_tokens)
    engine._ingestion_service = None
    engine._structured_ingestion = None
    engine._retrieval_namespace = None
    engine._ingestion_namespace = None
    engine._tree_indexing_service = None
    engine._tree_search_service = None
    return engine


async def test_query_mode_auto_routes_to_direct_when_corpus_below_threshold() -> None:
    """tokens=50_000 with threshold=150_000 → DIRECT path."""
    engine = _make_engine(threshold=150_000, corpus_tokens=50_000)
    engine._load_full_corpus = AsyncMock(return_value="corpus body")  # type: ignore[method-assign]

    await engine.query("q1", knowledge_id="kb-1")

    engine._knowledge_manager.get_corpus_tokens.assert_awaited_once_with("kb-1")
    engine._load_full_corpus.assert_awaited_once_with("kb-1")
    engine._generation_service.generate_from_corpus.assert_awaited_once()
    engine._retrieval_service.retrieve.assert_not_called()
    engine._generation_service.generate.assert_not_called()


async def test_query_mode_auto_routes_to_retrieval_when_corpus_above_threshold() -> None:
    """tokens=200_000 with threshold=150_000 → RETRIEVAL path."""
    engine = _make_engine(threshold=150_000, corpus_tokens=200_000)
    engine._load_full_corpus = AsyncMock(return_value="should not be called")  # type: ignore[method-assign]

    await engine.query("q1", knowledge_id="kb-1")

    engine._knowledge_manager.get_corpus_tokens.assert_awaited_once_with("kb-1")
    engine._retrieval_service.retrieve.assert_awaited_once()
    engine._generation_service.generate.assert_awaited_once()
    engine._load_full_corpus.assert_not_called()
    engine._generation_service.generate_from_corpus.assert_not_called()


async def test_query_mode_auto_uses_default_threshold_150k() -> None:
    """Default threshold is 150_000; boundary is `≤` for DIRECT."""
    # 149_999 → DIRECT (strictly below threshold)
    below = _make_engine(threshold=150_000, corpus_tokens=149_999)
    below._load_full_corpus = AsyncMock(return_value="corpus")  # type: ignore[method-assign]
    await below.query("q1", knowledge_id="kb-1")
    below._generation_service.generate_from_corpus.assert_awaited_once()
    below._generation_service.generate.assert_not_called()

    # 150_000 → DIRECT (equal: `tokens <= threshold`)
    at = _make_engine(threshold=150_000, corpus_tokens=150_000)
    at._load_full_corpus = AsyncMock(return_value="corpus")  # type: ignore[method-assign]
    await at.query("q1", knowledge_id="kb-1")
    at._generation_service.generate_from_corpus.assert_awaited_once()
    at._generation_service.generate.assert_not_called()

    # 150_001 → RETRIEVAL (strictly above threshold)
    above = _make_engine(threshold=150_000, corpus_tokens=150_001)
    above._load_full_corpus = AsyncMock(return_value="should not be called")  # type: ignore[method-assign]
    await above.query("q1", knowledge_id="kb-1")
    above._generation_service.generate.assert_awaited_once()
    above._generation_service.generate_from_corpus.assert_not_called()


async def test_query_mode_auto_respects_custom_threshold() -> None:
    """Custom threshold=50_000: 49_000 → DIRECT, 51_000 → RETRIEVAL."""
    below = _make_engine(threshold=50_000, corpus_tokens=49_000)
    below._load_full_corpus = AsyncMock(return_value="corpus")  # type: ignore[method-assign]
    await below.query("q1", knowledge_id="kb-1")
    below._generation_service.generate_from_corpus.assert_awaited_once()
    below._generation_service.generate.assert_not_called()

    above = _make_engine(threshold=50_000, corpus_tokens=51_000)
    above._load_full_corpus = AsyncMock(return_value="should not be called")  # type: ignore[method-assign]
    await above.query("q1", knowledge_id="kb-1")
    above._generation_service.generate.assert_awaited_once()
    above._generation_service.generate_from_corpus.assert_not_called()


async def test_query_mode_auto_routing_decision_in_trace_matches_chosen_branch() -> None:
    """trace.routing_decision is `"direct"` for DIRECT path, `"retrieval"` for RETRIEVAL path."""
    direct = _make_engine(threshold=150_000, corpus_tokens=10_000)
    direct._load_full_corpus = AsyncMock(return_value="corpus")  # type: ignore[method-assign]
    direct_result = await direct.query("q1", knowledge_id="kb-1", trace=True)
    assert direct_result.trace is not None
    assert direct_result.trace.routing_decision == "direct"

    retrieval = _make_engine(threshold=150_000, corpus_tokens=500_000)
    retrieval._retrieval_service.retrieve = AsyncMock(
        return_value=([], RetrievalTrace(query="q1", knowledge_id="kb-1"))
    )
    retrieval_result = await retrieval.query("q1", knowledge_id="kb-1", trace=True)
    assert retrieval_result.trace is not None
    assert retrieval_result.trace.routing_decision == "retrieval"


async def test_query_mode_auto_calls_get_corpus_tokens_once_per_query() -> None:
    """AUTO performs exactly one `get_corpus_tokens` read per query."""
    engine = _make_engine(threshold=150_000, corpus_tokens=10_000)
    engine._load_full_corpus = AsyncMock(return_value="corpus")  # type: ignore[method-assign]

    await engine.query("q1", knowledge_id="kb-1")

    assert engine._knowledge_manager.get_corpus_tokens.await_count == 1
