"""R5.3 — Confidence-based expansion + LC escalation.

Closes the R5 series with a self-healing retry loop wrapped around
`_retrieve_chunks` inside `_query_via_retrieval`. When the first attempt
returns weak chunks (`max(score) < grounding_threshold`), the loop retries
with `top_k *= 2` (capped at `top_k_max`); a second retry is reserved for
a future rewriter swap (no-op placeholder in R5.3). After
`max_expansion_retries` exhausted with chunks still weak, optional LC
escalation routes to `_query_via_direct_context` when the corpus fits the
direct-context threshold, otherwise proceeds with the weak chunks.

When `confidence_expansion=False` (default), the loop runs exactly once
and the `expansion_*` keys stay absent from `trace.adaptive` — distinct
from "ran with 0 retries" (those keys present, `expansion_attempts == 0`).

The retry loop lives at the engine layer (not the retrieval service):
engine has access to `KnowledgeManager.get_corpus_tokens` for the
LC-escalation decision and `_query_via_direct_context` for the actual
escalation. HYBRID's RAG branch calls `_retrieve_chunks` directly (not
`_query_via_retrieval`), so HYBRID is naturally excluded from expansion —
HYBRID has its own answerability check and would double up.

Bias-term hygiene: fixtures use neutral identifiers (`q1`, `chunk_a`, `kb-1`).
"""

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

from rfnry_rag.retrieval.common.formatting import ChunkOrdering
from rfnry_rag.retrieval.common.models import RetrievalTrace, RetrievedChunk
from rfnry_rag.retrieval.modules.generation.models import QueryResult
from rfnry_rag.retrieval.server import (
    AdaptiveRetrievalConfig,
    QueryMode,
    RagEngine,
    RagServerConfig,
    RoutingConfig,
)


def _chunk(chunk_id: str = "chunk_a", score: float = 0.9) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        source_id="src_a",
        content="some text",
        score=score,
    )


def _query_result(answer: str = "an answer") -> QueryResult:
    return QueryResult(answer=answer, sources=[], grounded=True, confidence=0.85)


def _make_engine(
    *,
    confidence_expansion: bool = True,
    max_expansion_retries: int = 2,
    top_k_min: int = 3,
    top_k_max: int = 15,
    base_top_k: int = 5,
    grounding_threshold: float = 0.5,
    direct_context_threshold: int = 150_000,
    corpus_tokens: int = 200_000,
) -> Any:
    """Build a minimally-wired RagEngine bypassing initialize().

    Returns `Any` so tests can poke `AsyncMock` assertion helpers on
    private service attributes typed as concrete services.
    """
    config = MagicMock(spec=RagServerConfig)
    config.retrieval = SimpleNamespace(
        history_window=3,
        top_k=base_top_k,
        adaptive=AdaptiveRetrievalConfig(
            enabled=True,
            top_k_min=top_k_min,
            top_k_max=top_k_max,
            confidence_expansion=confidence_expansion,
            max_expansion_retries=max_expansion_retries,
        ),
    )
    config.generation = SimpleNamespace(
        chunk_ordering=ChunkOrdering.SCORE_DESCENDING,
        grounding_threshold=grounding_threshold,
    )
    config.routing = RoutingConfig(
        mode=QueryMode.RETRIEVAL,
        direct_context_threshold=direct_context_threshold,
    )

    engine = RagEngine.__new__(RagEngine)
    engine._config = config
    engine._initialized = True
    engine._retrieval_service = AsyncMock()
    engine._structured_retrieval = None
    engine._generation_service = AsyncMock()
    cast(Any, engine._generation_service).generate = AsyncMock(return_value=_query_result())
    cast(Any, engine._generation_service).generate_from_corpus = AsyncMock(
        return_value=_query_result(answer="lc answer")
    )
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


async def test_expansion_disabled_runs_single_attempt() -> None:
    """`confidence_expansion=False` skips the retry loop entirely.

    Even when chunks are weak, the loop runs exactly once and the
    expansion keys (`expansion_attempts`, `expansion_outcome`,
    `final_top_k`) stay absent from `trace.adaptive` — keeping
    "didn't run" distinct from "ran with 0 retries".
    """
    engine = _make_engine(confidence_expansion=False)
    weak = [_chunk(score=0.1)]
    trace = RetrievalTrace(query="q1", knowledge_id="kb-1", adaptive={"foo": "bar"})
    engine._retrieve_chunks = AsyncMock(return_value=(weak, trace))  # type: ignore[method-assign]

    result = await engine.query("q1", knowledge_id="kb-1", trace=True)

    assert engine._retrieve_chunks.await_count == 1
    assert result.trace is not None
    assert result.trace.adaptive is not None
    # The R5.2 keys persist; the R5.3 keys are absent (loop didn't run).
    assert "expansion_attempts" not in result.trace.adaptive
    assert "expansion_outcome" not in result.trace.adaptive
    assert "final_top_k" not in result.trace.adaptive


async def test_expansion_succeeds_on_first_attempt_when_above_threshold() -> None:
    """First call returns strong chunks → no retry; outcome=succeeded, attempts=0.

    Locks the contract that even with 0 retries, `final_top_k` is reported
    (as `base_top_k`) — "didn't need to retry" is distinct from "didn't run
    expansion at all", and consumers can rely on the key being present.
    """
    engine = _make_engine(confidence_expansion=True, base_top_k=5, grounding_threshold=0.5)
    strong = [_chunk(score=0.8)]
    trace = RetrievalTrace(query="q1", knowledge_id="kb-1", adaptive={})
    engine._retrieve_chunks = AsyncMock(return_value=(strong, trace))  # type: ignore[method-assign]

    result = await engine.query("q1", knowledge_id="kb-1", trace=True)

    assert engine._retrieve_chunks.await_count == 1
    assert result.trace is not None
    assert result.trace.adaptive is not None
    assert result.trace.adaptive["expansion_attempts"] == 0
    assert result.trace.adaptive["expansion_outcome"] == "succeeded"
    assert result.trace.adaptive["final_top_k"] == 5


async def test_expansion_doubles_top_k_on_first_retry() -> None:
    """First call weak; retry with `top_k * 2` (capped at `top_k_max`).

    base_top_k=5 → first retry should pass top_k=10. Second call returns
    strong chunks so the loop exits cleanly.
    """
    engine = _make_engine(
        confidence_expansion=True,
        base_top_k=5,
        top_k_max=15,
        grounding_threshold=0.5,
    )
    weak = [_chunk(score=0.1)]
    strong = [_chunk(score=0.8)]
    trace1 = RetrievalTrace(query="q1", knowledge_id="kb-1", adaptive={})
    trace2 = RetrievalTrace(query="q1", knowledge_id="kb-1", adaptive={})
    engine._retrieve_chunks = AsyncMock(  # type: ignore[method-assign]
        side_effect=[(weak, trace1), (strong, trace2)]
    )

    result = await engine.query("q1", knowledge_id="kb-1", trace=True)

    assert engine._retrieve_chunks.await_count == 2
    second_call_kwargs = engine._retrieve_chunks.await_args_list[1].kwargs
    assert second_call_kwargs["top_k"] == 10
    assert result.trace is not None
    assert result.trace.adaptive is not None
    assert result.trace.adaptive["expansion_attempts"] == 1
    assert result.trace.adaptive["final_top_k"] == 10
    assert result.trace.adaptive["expansion_outcome"] == "succeeded"


async def test_expansion_exhausts_retries_then_proceeds_with_weak_chunks_when_corpus_too_large() -> None:
    """All retries weak + corpus > threshold → proceed with last weak chunks.

    `routing_decision` stays `"retrieval"` (not escalated). The returned
    chunks are the last (weak) attempt's chunks.
    """
    engine = _make_engine(
        confidence_expansion=True,
        max_expansion_retries=2,
        grounding_threshold=0.5,
        direct_context_threshold=150_000,
        corpus_tokens=200_000,
    )
    weak1 = [_chunk(chunk_id="weak_1", score=0.1)]
    weak2 = [_chunk(chunk_id="weak_2", score=0.2)]
    weak3 = [_chunk(chunk_id="weak_3", score=0.3)]
    trace1 = RetrievalTrace(query="q1", knowledge_id="kb-1", adaptive={})
    trace2 = RetrievalTrace(query="q1", knowledge_id="kb-1", adaptive={})
    trace3 = RetrievalTrace(query="q1", knowledge_id="kb-1", adaptive={})
    engine._retrieve_chunks = AsyncMock(  # type: ignore[method-assign]
        side_effect=[(weak1, trace1), (weak2, trace2), (weak3, trace3)]
    )

    result = await engine.query("q1", knowledge_id="kb-1", trace=True)

    # 1 first attempt + 2 retries = 3 calls.
    assert engine._retrieve_chunks.await_count == 3
    engine._generation_service.generate.assert_awaited_once()
    engine._generation_service.generate_from_corpus.assert_not_called()
    assert result.trace is not None
    assert result.trace.adaptive is not None
    assert result.trace.adaptive["expansion_attempts"] == 2
    assert result.trace.adaptive["expansion_outcome"] == "exhausted_proceeded"
    assert result.trace.routing_decision == "retrieval"


async def test_expansion_escalates_to_direct_when_corpus_below_threshold() -> None:
    """All retries weak + corpus ≤ threshold → escalate to DIRECT.

    `_query_via_direct_context` is invoked; the returned answer comes
    from the DIRECT path (`generate_from_corpus`), not from the weak
    chunks (`generate`).
    """
    engine = _make_engine(
        confidence_expansion=True,
        max_expansion_retries=2,
        grounding_threshold=0.5,
        direct_context_threshold=150_000,
        corpus_tokens=50_000,
    )
    weak = [_chunk(score=0.1)]
    trace = RetrievalTrace(query="q1", knowledge_id="kb-1", adaptive={})
    engine._retrieve_chunks = AsyncMock(  # type: ignore[method-assign]
        side_effect=[(weak, trace), (weak, trace), (weak, trace)]
    )
    engine._load_full_corpus = AsyncMock(return_value="corpus body")  # type: ignore[method-assign]

    result = await engine.query("q1", knowledge_id="kb-1", trace=True)

    engine._load_full_corpus.assert_awaited_once_with("kb-1")
    engine._generation_service.generate_from_corpus.assert_awaited_once()
    engine._generation_service.generate.assert_not_called()
    assert result.answer == "lc answer"


async def test_expansion_records_attempts_and_outcome_in_trace() -> None:
    """One retry then success: attempts=1, outcome=succeeded, final_top_k=10."""
    engine = _make_engine(
        confidence_expansion=True,
        base_top_k=5,
        top_k_max=15,
        grounding_threshold=0.5,
    )
    weak = [_chunk(score=0.1)]
    strong = [_chunk(score=0.8)]
    trace1 = RetrievalTrace(query="q1", knowledge_id="kb-1", adaptive={})
    trace2 = RetrievalTrace(query="q1", knowledge_id="kb-1", adaptive={})
    engine._retrieve_chunks = AsyncMock(  # type: ignore[method-assign]
        side_effect=[(weak, trace1), (strong, trace2)]
    )

    result = await engine.query("q1", knowledge_id="kb-1", trace=True)

    assert result.trace is not None
    assert result.trace.adaptive is not None
    assert result.trace.adaptive["expansion_attempts"] == 1
    assert result.trace.adaptive["expansion_outcome"] == "succeeded"
    assert result.trace.adaptive["final_top_k"] == 10


async def test_expansion_escalation_routing_decision_is_retrieval_then_direct() -> None:
    """LC escalation sets the new `retrieval_then_direct` routing_decision value.

    Distinguishes "AUTO chose DIRECT directly" (= "direct") from
    "RETRIEVAL ran, expansion failed, escalated" (= "retrieval_then_direct").
    Different cost shape, different debugging signal.
    """
    engine = _make_engine(
        confidence_expansion=True,
        max_expansion_retries=2,
        grounding_threshold=0.5,
        direct_context_threshold=150_000,
        corpus_tokens=50_000,
    )
    weak = [_chunk(score=0.1)]
    trace = RetrievalTrace(query="q1", knowledge_id="kb-1", adaptive={})
    engine._retrieve_chunks = AsyncMock(  # type: ignore[method-assign]
        side_effect=[(weak, trace), (weak, trace), (weak, trace)]
    )
    engine._load_full_corpus = AsyncMock(return_value="corpus body")  # type: ignore[method-assign]

    result = await engine.query("q1", knowledge_id="kb-1", trace=True)

    assert result.trace is not None
    assert result.trace.routing_decision == "retrieval_then_direct"


async def test_expansion_uses_grounding_threshold_from_generation_config() -> None:
    """Threshold source is `GenerationConfig.grounding_threshold` (single source of truth).

    threshold=0.6: chunks at 0.5 trigger expansion (weak); chunks at 0.7
    skip expansion (strong). The boundary is `<` not `<=` — a query at
    exactly the threshold is NOT considered weak.
    """
    # Below threshold → expand.
    weak_engine = _make_engine(
        confidence_expansion=True,
        grounding_threshold=0.6,
    )
    weak = [_chunk(score=0.5)]
    strong = [_chunk(score=0.7)]
    trace1 = RetrievalTrace(query="q1", knowledge_id="kb-1", adaptive={})
    trace2 = RetrievalTrace(query="q1", knowledge_id="kb-1", adaptive={})
    weak_engine._retrieve_chunks = AsyncMock(  # type: ignore[method-assign]
        side_effect=[(weak, trace1), (strong, trace2)]
    )
    await weak_engine.query("q1", knowledge_id="kb-1", trace=True)
    assert weak_engine._retrieve_chunks.await_count == 2

    # Above threshold → no expansion.
    strong_engine = _make_engine(
        confidence_expansion=True,
        grounding_threshold=0.6,
    )
    trace3 = RetrievalTrace(query="q1", knowledge_id="kb-1", adaptive={})
    strong_engine._retrieve_chunks = AsyncMock(  # type: ignore[method-assign]
        return_value=([_chunk(score=0.7)], trace3)
    )
    await strong_engine.query("q1", knowledge_id="kb-1", trace=True)
    assert strong_engine._retrieve_chunks.await_count == 1


async def test_expansion_escalation_preserves_r5_2_adaptive_classification_keys() -> None:
    """LC escalation MERGES the pre-escalation classifier verdict onto the DIRECT trace.

    A consumer debugging "why did this escalate?" needs both signals:

    - R5.2's classifier verdict (`complexity`, `query_type`,
      `effective_top_k`, `applied_multipliers`, `classification_source`)
      explains what the classifier said about the failed retrieval.
    - R5.3's expansion keys (`expansion_attempts`, `expansion_outcome`,
      `final_top_k`) explain why the engine gave up on RAG and escalated.

    Without the merge, the classifier verdict would be silently dropped at
    the escalation boundary because `_query_via_direct_context` returns a
    fresh trace whose `adaptive` is `None`.
    """
    engine = _make_engine(
        confidence_expansion=True,
        max_expansion_retries=2,
        grounding_threshold=0.5,
        direct_context_threshold=150_000,
        corpus_tokens=50_000,
    )
    weak = [_chunk(score=0.1)]
    # Classifier verdict the failed retrieval was based on — R5.2's adaptive
    # block carries these keys; the merge must preserve them.
    pre_escalation_adaptive = {
        "complexity": "COMPLEX",
        "query_type": "ENTITY_RELATIONSHIP",
        "effective_top_k": 15,
        "applied_multipliers": {"vector": 0.8, "graph": 1.5},
        "classification_source": "heuristic",
    }
    trace_with_classification = RetrievalTrace(
        query="q1",
        knowledge_id="kb-1",
        adaptive=dict(pre_escalation_adaptive),
    )
    engine._retrieve_chunks = AsyncMock(  # type: ignore[method-assign]
        side_effect=[
            (weak, trace_with_classification),
            (weak, trace_with_classification),
            (weak, trace_with_classification),
        ]
    )
    engine._load_full_corpus = AsyncMock(return_value="corpus body")  # type: ignore[method-assign]

    result = await engine.query("q1", knowledge_id="kb-1", trace=True)

    assert result.trace is not None
    assert result.trace.adaptive is not None
    # R5.2 keys preserved across the escalation boundary.
    assert result.trace.adaptive["complexity"] == "COMPLEX"
    assert result.trace.adaptive["query_type"] == "ENTITY_RELATIONSHIP"
    assert result.trace.adaptive["effective_top_k"] == 15
    assert result.trace.adaptive["applied_multipliers"] == {"vector": 0.8, "graph": 1.5}
    assert result.trace.adaptive["classification_source"] == "heuristic"
    # R5.3 keys layered on top.
    assert result.trace.adaptive["expansion_outcome"] == "exhausted_escalated_to_lc"
    assert result.trace.adaptive["expansion_attempts"] == 2
    assert "final_top_k" in result.trace.adaptive
    # Routing decision distinguishes RAG-then-LC from plain DIRECT.
    assert result.trace.routing_decision == "retrieval_then_direct"


async def test_expansion_escalation_preserves_pre_escalation_timings() -> None:
    """LC escalation MERGES pre-escalation RAG timings onto the DIRECT trace.

    `routing_decision="retrieval_then_direct"` exists specifically to flag
    the RAG-then-LC cost shape for debugging consumers. If only DIRECT-stage
    timings (`direct_context_load`, `generation`) survive the escalation,
    the cost-shape attribution is broken — consumers can't see the
    rewriting / retrieval / fusion / reranking / classification / retry
    overhead that was paid before the escalation.

    The merge layers RAG timings UNDER DIRECT timings (DIRECT wins on
    key collision; none expected since RAG and DIRECT stage names are
    distinct). Both sets must be present in the final trace.
    """
    engine = _make_engine(
        confidence_expansion=True,
        max_expansion_retries=2,
        grounding_threshold=0.5,
        direct_context_threshold=150_000,
        corpus_tokens=50_000,
    )
    weak = [_chunk(score=0.1)]
    # Pre-escalation RAG timings — what the retrieval pipeline accumulated
    # before the engine gave up and escalated to DIRECT.
    pre_escalation_timings = {
        "retrieval": 0.05,
        "fusion": 0.01,
        "classification": 0.001,
    }
    trace_with_timings = RetrievalTrace(
        query="q1",
        knowledge_id="kb-1",
        adaptive={},
    )
    trace_with_timings.timings.update(pre_escalation_timings)
    engine._retrieve_chunks = AsyncMock(  # type: ignore[method-assign]
        side_effect=[
            (weak, trace_with_timings),
            (weak, trace_with_timings),
            (weak, trace_with_timings),
        ]
    )
    engine._load_full_corpus = AsyncMock(return_value="corpus body")  # type: ignore[method-assign]

    # Patch `_query_via_direct_context` to return a result whose trace
    # carries known DIRECT-stage timings — this is what the escalation
    # path attaches.
    direct_trace = RetrievalTrace(
        query="q1",
        knowledge_id="kb-1",
        routing_decision="direct",
    )
    direct_trace.timings.update({"direct_context_load": 0.02, "generation": 1.5})
    direct_qr = _query_result(answer="lc answer")
    direct_qr.trace = direct_trace
    engine._query_via_direct_context = AsyncMock(return_value=direct_qr)  # type: ignore[method-assign]

    result = await engine.query("q1", knowledge_id="kb-1", trace=True)

    assert result.trace is not None
    # Both RAG-stage and DIRECT-stage timings must be present so consumers
    # can attribute the full RAG-then-LC cost shape.
    assert "retrieval" in result.trace.timings
    assert "fusion" in result.trace.timings
    assert "classification" in result.trace.timings
    assert "direct_context_load" in result.trace.timings
    assert "generation" in result.trace.timings
    # Values are preserved (not zeroed / overwritten).
    assert result.trace.timings["retrieval"] == 0.05
    assert result.trace.timings["fusion"] == 0.01
    assert result.trace.timings["classification"] == 0.001
    assert result.trace.timings["direct_context_load"] == 0.02
    assert result.trace.timings["generation"] == 1.5
