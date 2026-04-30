"""AUTO mode: token-count-based dispatch between RETRIEVAL and DIRECT.

`mode="auto"` is user-facing and recommended for new users. AUTO reads
`KnowledgeManager.get_corpus_tokens(knowledge_id)` and routes to DIRECT when
`tokens <= full_context_threshold`, otherwise RETRIEVAL.
"""

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

from rfnry_rag.config import QueryMode, RoutingConfig
from rfnry_rag.observability.trace import RetrievalTrace


def _engine(
    make_engine: Any,
    *,
    mode: QueryMode = QueryMode.AUTO,
    threshold: int = 150_000,
    corpus_tokens: int = 0,
) -> Any:
    """Thin shim: build a knowledge_manager mock and forward to ``make_engine``.

    Each AUTO-mode test needs a ``knowledge_manager.get_corpus_tokens`` that
    returns a configurable token count; the factory's default leaves the
    manager as ``None``.
    """
    km = MagicMock()
    cast(Any, km).get_corpus_tokens = AsyncMock(return_value=corpus_tokens)
    return make_engine(
        retrieval=SimpleNamespace(history_window=3),
        routing=RoutingConfig(mode=mode, full_context_threshold=threshold),
        knowledge_manager=km,
    )


async def test_query_mode_auto_routes_to_direct_when_corpus_below_threshold(
    make_engine: Any,
) -> None:
    """tokens=50_000 with threshold=150_000 → DIRECT path."""
    engine = _engine(make_engine, threshold=150_000, corpus_tokens=50_000)
    engine._load_full_corpus = AsyncMock(return_value="corpus body")  # type: ignore[method-assign]

    await engine.query("q1", knowledge_id="kb-1")

    engine._knowledge_manager.get_corpus_tokens.assert_awaited_once_with("kb-1")
    engine._load_full_corpus.assert_awaited_once_with("kb-1")
    engine._generation_service.generate_from_corpus.assert_awaited_once()
    engine._retrieval_service.retrieve.assert_not_called()
    engine._generation_service.generate.assert_not_called()


async def test_query_mode_auto_routes_to_retrieval_when_corpus_above_threshold(
    make_engine: Any,
) -> None:
    """tokens=200_000 with threshold=150_000 → RETRIEVAL path."""
    engine = _engine(make_engine, threshold=150_000, corpus_tokens=200_000)
    engine._load_full_corpus = AsyncMock(return_value="should not be called")  # type: ignore[method-assign]

    await engine.query("q1", knowledge_id="kb-1")

    engine._knowledge_manager.get_corpus_tokens.assert_awaited_once_with("kb-1")
    engine._retrieval_service.retrieve.assert_awaited_once()
    engine._generation_service.generate.assert_awaited_once()
    engine._load_full_corpus.assert_not_called()
    engine._generation_service.generate_from_corpus.assert_not_called()


async def test_query_mode_auto_uses_default_threshold_150k(make_engine: Any) -> None:
    """Default threshold is 150_000; boundary is `≤` for DIRECT."""
    # 149_999 → DIRECT (strictly below threshold)
    below = _engine(make_engine, threshold=150_000, corpus_tokens=149_999)
    below._load_full_corpus = AsyncMock(return_value="corpus")  # type: ignore[method-assign]
    await below.query("q1", knowledge_id="kb-1")
    below._generation_service.generate_from_corpus.assert_awaited_once()
    below._generation_service.generate.assert_not_called()

    # 150_000 → DIRECT (equal: `tokens <= threshold`)
    at = _engine(make_engine, threshold=150_000, corpus_tokens=150_000)
    at._load_full_corpus = AsyncMock(return_value="corpus")  # type: ignore[method-assign]
    await at.query("q1", knowledge_id="kb-1")
    at._generation_service.generate_from_corpus.assert_awaited_once()
    at._generation_service.generate.assert_not_called()

    # 150_001 → RETRIEVAL (strictly above threshold)
    above = _engine(make_engine, threshold=150_000, corpus_tokens=150_001)
    above._load_full_corpus = AsyncMock(return_value="should not be called")  # type: ignore[method-assign]
    await above.query("q1", knowledge_id="kb-1")
    above._generation_service.generate.assert_awaited_once()
    above._generation_service.generate_from_corpus.assert_not_called()


async def test_query_mode_auto_respects_custom_threshold(make_engine: Any) -> None:
    """Custom threshold=50_000: 49_000 → DIRECT, 51_000 → RETRIEVAL."""
    below = _engine(make_engine, threshold=50_000, corpus_tokens=49_000)
    below._load_full_corpus = AsyncMock(return_value="corpus")  # type: ignore[method-assign]
    await below.query("q1", knowledge_id="kb-1")
    below._generation_service.generate_from_corpus.assert_awaited_once()
    below._generation_service.generate.assert_not_called()

    above = _engine(make_engine, threshold=50_000, corpus_tokens=51_000)
    above._load_full_corpus = AsyncMock(return_value="should not be called")  # type: ignore[method-assign]
    await above.query("q1", knowledge_id="kb-1")
    above._generation_service.generate.assert_awaited_once()
    above._generation_service.generate_from_corpus.assert_not_called()


async def test_query_mode_auto_routing_decision_in_trace_matches_chosen_branch(
    make_engine: Any,
) -> None:
    """trace.routing_decision is `"direct"` for DIRECT path, `"retrieval"` for RETRIEVAL path."""
    direct = _engine(make_engine, threshold=150_000, corpus_tokens=10_000)
    direct._load_full_corpus = AsyncMock(return_value="corpus")  # type: ignore[method-assign]
    direct_result = await direct.query("q1", knowledge_id="kb-1", trace=True)
    assert direct_result.trace is not None
    assert direct_result.trace.routing_decision == "full_context"

    retrieval = _engine(make_engine, threshold=150_000, corpus_tokens=500_000)
    retrieval._retrieval_service.retrieve = AsyncMock(
        return_value=([], RetrievalTrace(query="q1", knowledge_id="kb-1"))
    )
    retrieval_result = await retrieval.query("q1", knowledge_id="kb-1", trace=True)
    assert retrieval_result.trace is not None
    assert retrieval_result.trace.routing_decision == "indexed"


async def test_query_mode_auto_calls_get_corpus_tokens_once_per_query(
    make_engine: Any,
) -> None:
    """AUTO performs exactly one `get_corpus_tokens` read per query."""
    engine = _engine(make_engine, threshold=150_000, corpus_tokens=10_000)
    engine._load_full_corpus = AsyncMock(return_value="corpus")  # type: ignore[method-assign]

    await engine.query("q1", knowledge_id="kb-1")

    assert engine._knowledge_manager.get_corpus_tokens.await_count == 1
