"""R6.2 — IterativeRetrievalService hop loop + engine arm.

Lights up the runtime core of multi-hop iterative retrieval. R6.1 shipped
the config + BAML scaffold; R6.2 lands the loop, the dedup helper, the
engine `_query_via_iterative` arm, and the trace-surface population. R6.3
will add post-loop DIRECT escalation.

Bias-term hygiene: fixtures use neutral identifiers (`q1`, `chunk_a`,
`topic_a`, `kb-1`). No domain-specific vocabulary.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rfnry_rag.retrieval.common.errors import ConfigurationError
from rfnry_rag.retrieval.common.formatting import ChunkOrdering
from rfnry_rag.retrieval.common.language_model import LanguageModelClient, LanguageModelProvider
from rfnry_rag.retrieval.common.models import RetrievalTrace, RetrievedChunk
from rfnry_rag.retrieval.modules.generation.models import QueryResult
from rfnry_rag.retrieval.modules.retrieval.iterative.config import IterativeRetrievalConfig
from rfnry_rag.retrieval.modules.retrieval.iterative.service import (
    IterativeRetrievalService,
    _merge_chunks_dedup,
)
from rfnry_rag.retrieval.modules.retrieval.search.classification import (
    QueryClassification,
    QueryComplexity,
    QueryType,
)
from rfnry_rag.retrieval.server import (
    AdaptiveRetrievalConfig,
    QueryMode,
    RagEngine,
    RagServerConfig,
    RetrievalConfig,
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


def _classification(
    complexity: QueryComplexity = QueryComplexity.MODERATE,
    query_type: QueryType = QueryType.FACTUAL,
) -> QueryClassification:
    return QueryClassification(
        complexity=complexity,
        query_type=query_type,
        signals={},
        source="heuristic",
    )


def _decompose_result(
    done: bool,
    next_sub_question: str | None = None,
    findings: str = "",
    reasoning: str = "r",
) -> SimpleNamespace:
    """Light stand-in for `baml_client.types.DecomposeResult`.

    The service reads four attributes (`done`, `next_sub_question`,
    `findings_from_last_hop`, `reasoning`); a SimpleNamespace is enough.
    """
    return SimpleNamespace(
        done=done,
        next_sub_question=next_sub_question,
        findings_from_last_hop=findings,
        reasoning=reasoning,
    )


def _lm_client() -> LanguageModelClient:
    return LanguageModelClient(
        provider=LanguageModelProvider(provider="openai", model="gpt-test", api_key="x"),
    )


def _make_engine(
    *,
    iterative_enabled: bool = True,
    gate_mode: str = "type",
    max_hops: int = 3,
) -> Any:
    """Build a minimally-wired RagEngine bypassing initialize().

    Mirrors R5.3's test pattern: real `RetrievalConfig` (so `iterative` and
    `adaptive` exist), `MagicMock` for the dependency services, `AsyncMock`
    return values where awaitable.
    """
    iterative_cfg = IterativeRetrievalConfig(
        enabled=iterative_enabled,
        gate_mode=gate_mode,
        max_hops=max_hops,
        decomposition_model=_lm_client() if gate_mode == "llm" else None,
    )
    config = MagicMock(spec=RagServerConfig)
    config.retrieval = RetrievalConfig(
        top_k=5,
        adaptive=AdaptiveRetrievalConfig(enabled=False),
        iterative=iterative_cfg,
        enrich_lm_client=_lm_client(),
    )
    config.generation = SimpleNamespace(
        chunk_ordering=ChunkOrdering.SCORE_DESCENDING,
        grounding_threshold=0.5,
    )
    config.routing = RoutingConfig(
        mode=QueryMode.RETRIEVAL,
        direct_context_threshold=150_000,
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
    cast(Any, engine._knowledge_manager).get_corpus_tokens = AsyncMock(return_value=200_000)
    engine._ingestion_service = None
    engine._structured_ingestion = None
    engine._retrieval_namespace = None
    engine._ingestion_namespace = None
    engine._tree_indexing_service = None
    engine._tree_search_service = None
    engine._iterative_service = (
        IterativeRetrievalService(
            retrieval_service=engine._retrieval_service,
            fallback_decomposition_lm=_lm_client(),
        )
        if iterative_enabled
        else None
    )
    return engine


_DECOMPOSE_PATH = "rfnry_rag.retrieval.modules.retrieval.iterative.service.b.DecomposeQuery"
_BUILD_REGISTRY_PATH = (
    "rfnry_rag.retrieval.modules.retrieval.iterative.service.build_registry"
)
_CLASSIFY_PATH = "rfnry_rag.retrieval.server.classify_query"


# ---------------------------------------------------------------------------
# Test 1: iterative disabled falls through to plain retrieval.
# ---------------------------------------------------------------------------


async def test_iterative_disabled_falls_through_to_retrieval() -> None:
    """With `iterative.enabled=False`, `_query_via_iterative` is never called."""
    engine = _make_engine(iterative_enabled=False)
    chunks = [_chunk()]
    trace = RetrievalTrace(query="q1", knowledge_id="kb-1")
    engine._retrieve_chunks = AsyncMock(return_value=(chunks, trace))  # type: ignore[method-assign]
    engine._query_via_iterative = AsyncMock()  # type: ignore[method-assign]

    result = await engine.query("q1", knowledge_id="kb-1", trace=True)

    cast(Any, engine._query_via_iterative).assert_not_called()
    assert result.trace is not None
    assert result.trace.routing_decision == "retrieval"


# ---------------------------------------------------------------------------
# Test 2: type-mode gate fails on simple/factual -> falls through to retrieval.
# ---------------------------------------------------------------------------


async def test_iterative_type_gate_fails_falls_through_to_retrieval() -> None:
    """Type-mode + SIMPLE/FACTUAL classification -> plain retrieval path."""
    engine = _make_engine(gate_mode="type")
    chunks = [_chunk()]
    trace = RetrievalTrace(query="q1", knowledge_id="kb-1")
    engine._retrieve_chunks = AsyncMock(return_value=(chunks, trace))  # type: ignore[method-assign]
    engine._query_via_iterative = AsyncMock()  # type: ignore[method-assign]

    with patch(
        _CLASSIFY_PATH,
        new=AsyncMock(
            return_value=_classification(
                complexity=QueryComplexity.SIMPLE, query_type=QueryType.FACTUAL
            )
        ),
    ):
        result = await engine.query("q1", knowledge_id="kb-1", trace=True)

    cast(Any, engine._query_via_iterative).assert_not_called()
    assert result.trace is not None
    assert result.trace.routing_decision == "retrieval"


# ---------------------------------------------------------------------------
# Test 3: type-mode gate passes on entity-relationship -> iterative runs.
# ---------------------------------------------------------------------------


async def test_iterative_type_gate_passes_on_entity_relationship() -> None:
    """ENTITY_RELATIONSHIP classification triggers `_query_via_iterative`."""
    engine = _make_engine(gate_mode="type")
    engine._query_via_iterative = AsyncMock(return_value=_query_result())  # type: ignore[method-assign]

    with patch(
        _CLASSIFY_PATH,
        new=AsyncMock(
            return_value=_classification(
                complexity=QueryComplexity.MODERATE,
                query_type=QueryType.ENTITY_RELATIONSHIP,
            )
        ),
    ):
        await engine.query("q1", knowledge_id="kb-1", trace=False)

    cast(Any, engine._query_via_iterative).assert_called_once()


# ---------------------------------------------------------------------------
# Test 4: type-mode gate passes on COMPLEX -> iterative runs.
# ---------------------------------------------------------------------------


async def test_iterative_type_gate_passes_on_complex() -> None:
    """COMPLEX/FACTUAL classification triggers `_query_via_iterative`."""
    engine = _make_engine(gate_mode="type")
    engine._query_via_iterative = AsyncMock(return_value=_query_result())  # type: ignore[method-assign]

    with patch(
        _CLASSIFY_PATH,
        new=AsyncMock(
            return_value=_classification(
                complexity=QueryComplexity.COMPLEX, query_type=QueryType.FACTUAL
            )
        ),
    ):
        await engine.query("q1", knowledge_id="kb-1", trace=False)

    cast(Any, engine._query_via_iterative).assert_called_once()


# ---------------------------------------------------------------------------
# Test 5: LLM-mode + first decompose returns done -> short-circuits.
# ---------------------------------------------------------------------------


async def test_iterative_llm_gate_first_decompose_says_done_short_circuits() -> None:
    """LLM-mode: `done=true` on hop 0 -> 1 decompose, 0 retrieves."""
    engine = _make_engine(gate_mode="llm")
    cast(Any, engine._retrieval_service).retrieve = AsyncMock(
        return_value=([_chunk()], RetrievalTrace(query="x"))
    )

    with (
        patch(_BUILD_REGISTRY_PATH, return_value=MagicMock()),
        patch(
            _DECOMPOSE_PATH,
            new=AsyncMock(return_value=_decompose_result(done=True, findings="all known")),
        ) as mock_decompose,
    ):
        result = await engine.query("q1", knowledge_id="kb-1", trace=True)

    assert mock_decompose.await_count == 1
    cast(Any, engine._retrieval_service).retrieve.assert_not_awaited()
    assert result.trace is not None
    assert result.trace.routing_decision == "iterative"
    assert result.trace.iterative_termination_reason == "done"
    assert result.trace.iterative_hops is not None
    assert len(result.trace.iterative_hops) == 1
    assert result.trace.iterative_hops[0].sub_question is None


# ---------------------------------------------------------------------------
# Test 6: LLM-mode + first decompose proceeds, second is done.
# ---------------------------------------------------------------------------


async def test_iterative_llm_gate_first_decompose_proceeds() -> None:
    """LLM-mode: hop 0 yields a sub_question, hop 1 says done."""
    engine = _make_engine(gate_mode="llm")
    cast(Any, engine._retrieval_service).retrieve = AsyncMock(
        return_value=([_chunk("chunk_a", 0.9)], RetrievalTrace(query="x"))
    )

    decompose_mock = AsyncMock(
        side_effect=[
            _decompose_result(
                done=False, next_sub_question="topic_a sub-question", findings="hop0 findings"
            ),
            _decompose_result(done=True, findings="hop1 findings"),
        ]
    )

    with (
        patch(_BUILD_REGISTRY_PATH, return_value=MagicMock()),
        patch(_DECOMPOSE_PATH, new=decompose_mock),
    ):
        result = await engine.query("q1", knowledge_id="kb-1", trace=True)

    assert decompose_mock.await_count == 2
    assert cast(Any, engine._retrieval_service).retrieve.await_count == 1
    assert result.trace is not None
    assert result.trace.iterative_termination_reason == "done"
    assert result.trace.iterative_hops is not None
    assert len(result.trace.iterative_hops) == 2
    assert result.trace.iterative_hops[0].sub_question == "topic_a sub-question"
    assert result.trace.iterative_hops[1].sub_question is None


# ---------------------------------------------------------------------------
# Test 7: max_hops reached without done -> termination_reason="max_hops".
# ---------------------------------------------------------------------------


async def test_iterative_max_hops_termination() -> None:
    """All decompose calls return done=false -> exit at max_hops."""
    engine = _make_engine(gate_mode="llm", max_hops=3)
    cast(Any, engine._retrieval_service).retrieve = AsyncMock(
        return_value=([_chunk("chunk_a", 0.9)], RetrievalTrace(query="x"))
    )

    decompose_mock = AsyncMock(
        return_value=_decompose_result(
            done=False, next_sub_question="next sub-q", findings="findings"
        )
    )

    with (
        patch(_BUILD_REGISTRY_PATH, return_value=MagicMock()),
        patch(_DECOMPOSE_PATH, new=decompose_mock),
    ):
        result = await engine.query("q1", knowledge_id="kb-1", trace=True)

    assert decompose_mock.await_count == 3
    assert cast(Any, engine._retrieval_service).retrieve.await_count == 3
    assert result.trace is not None
    assert result.trace.iterative_termination_reason == "max_hops"
    assert result.trace.iterative_hops is not None
    assert len(result.trace.iterative_hops) == 3


# ---------------------------------------------------------------------------
# Test 8: dedup keeps higher score on collision.
# ---------------------------------------------------------------------------


def test_iterative_dedup_preserves_higher_score_on_collision() -> None:
    """Same chunk_id seen twice -> higher score wins, slot updates in place."""
    accumulated = [_chunk("chunk_a", 0.5), _chunk("chunk_b", 0.6)]
    new = [_chunk("chunk_a", 0.7), _chunk("chunk_c", 0.4)]
    merged = _merge_chunks_dedup(accumulated, new)

    assert len(merged) == 3
    # Order: existing slots stay; collision updates in place; new chunks append.
    assert [c.chunk_id for c in merged] == ["chunk_a", "chunk_b", "chunk_c"]
    assert merged[0].score == 0.7  # higher score kept
    assert merged[1].score == 0.6  # untouched
    assert merged[2].score == 0.4  # appended

    # Lower-score collision must NOT downgrade the existing slot.
    accumulated2 = [_chunk("chunk_a", 0.9)]
    new2 = [_chunk("chunk_a", 0.1)]
    merged2 = _merge_chunks_dedup(accumulated2, new2)
    assert len(merged2) == 1
    assert merged2[0].score == 0.9


# ---------------------------------------------------------------------------
# Test 9: findings flow from hop N's decompose result to hop N+1's input.
# ---------------------------------------------------------------------------


async def test_iterative_findings_passed_to_next_decompose_call() -> None:
    """Hop 1's `accumulated_findings` arg == hop 0's returned `findings_from_last_hop`."""
    engine = _make_engine(gate_mode="llm")
    cast(Any, engine._retrieval_service).retrieve = AsyncMock(
        return_value=([_chunk()], RetrievalTrace(query="x"))
    )

    hop0_findings = "hop 0 produced findings about topic_a"
    decompose_mock = AsyncMock(
        side_effect=[
            _decompose_result(
                done=False, next_sub_question="next sub-q", findings=hop0_findings
            ),
            _decompose_result(done=True, findings="final summary"),
        ]
    )

    with (
        patch(_BUILD_REGISTRY_PATH, return_value=MagicMock()),
        patch(_DECOMPOSE_PATH, new=decompose_mock),
    ):
        await engine.query("q1", knowledge_id="kb-1", trace=False)

    # Hop 0's call: empty findings on entry.
    call0 = decompose_mock.await_args_list[0]
    assert call0.kwargs["accumulated_findings"] == ""
    # Hop 1's call: hop 0's returned findings.
    call1 = decompose_mock.await_args_list[1]
    assert call1.kwargs["accumulated_findings"] == hop0_findings


# ---------------------------------------------------------------------------
# Test 10: trace populates the hop list with ordered sub-questions.
# ---------------------------------------------------------------------------


async def test_iterative_trace_populates_hop_list() -> None:
    """`trace=True` -> `iterative_hops` populated with ordered hop traces."""
    engine = _make_engine(gate_mode="llm", max_hops=4)
    # Per-hop trace carries R5.2's adaptive verdict — assert it lands in the
    # IterativeHopTrace.adaptive field (boundary-preservation guard).
    inner_trace = RetrievalTrace(
        query="x",
        adaptive={
            "complexity": "COMPLEX",
            "query_type": "FACTUAL",
            "effective_top_k": 10,
            "applied_multipliers": {"vector": 1.2},
            "classification_source": "heuristic",
        },
    )
    cast(Any, engine._retrieval_service).retrieve = AsyncMock(
        return_value=([_chunk("chunk_a", 0.9)], inner_trace)
    )

    decompose_mock = AsyncMock(
        side_effect=[
            _decompose_result(
                done=False, next_sub_question="sub-q-1", findings="f1"
            ),
            _decompose_result(
                done=False, next_sub_question="sub-q-2", findings="f2"
            ),
            _decompose_result(done=True, findings="f3"),
        ]
    )

    with (
        patch(_BUILD_REGISTRY_PATH, return_value=MagicMock()),
        patch(_DECOMPOSE_PATH, new=decompose_mock),
    ):
        result = await engine.query("q1", knowledge_id="kb-1", trace=True)

    assert result.trace is not None
    hops = result.trace.iterative_hops
    assert hops is not None
    assert len(hops) == 3
    assert [h.sub_question for h in hops] == ["sub-q-1", "sub-q-2", None]
    # The done-call has no sub_question; first two are real hops.
    assert hops[2].sub_question is None
    # R5.2 adaptive verdict survived the boundary into the per-hop trace.
    assert hops[0].adaptive is not None
    assert hops[0].adaptive["complexity"] == "COMPLEX"
    assert hops[0].adaptive["applied_multipliers"] == {"vector": 1.2}


# ---------------------------------------------------------------------------
# Test 11: routing_decision is "iterative" on a successful run.
# ---------------------------------------------------------------------------


async def test_iterative_routing_decision_is_iterative() -> None:
    """A run that takes the iterative arm carries `routing_decision="iterative"`."""
    engine = _make_engine(gate_mode="llm")
    cast(Any, engine._retrieval_service).retrieve = AsyncMock(
        return_value=([_chunk()], RetrievalTrace(query="x"))
    )

    with (
        patch(_BUILD_REGISTRY_PATH, return_value=MagicMock()),
        patch(
            _DECOMPOSE_PATH,
            new=AsyncMock(return_value=_decompose_result(done=True, findings="done")),
        ),
    ):
        result = await engine.query("q1", knowledge_id="kb-1", trace=True)

    assert result.trace is not None
    assert result.trace.routing_decision == "iterative"


# ---------------------------------------------------------------------------
# Test 12: contract violation (done=false but null sub_question) terminates.
# ---------------------------------------------------------------------------


async def test_iterative_decomposer_contract_violation_terminates_with_error() -> None:
    """`done=false, next_sub_question=None` -> termination_reason="error", no retrieve."""
    engine = _make_engine(gate_mode="llm")
    cast(Any, engine._retrieval_service).retrieve = AsyncMock()

    with (
        patch(_BUILD_REGISTRY_PATH, return_value=MagicMock()),
        patch(
            _DECOMPOSE_PATH,
            new=AsyncMock(
                return_value=_decompose_result(
                    done=False, next_sub_question=None, findings="partial"
                )
            ),
        ),
    ):
        result = await engine.query("q1", knowledge_id="kb-1", trace=True)

    cast(Any, engine._retrieval_service).retrieve.assert_not_awaited()
    assert result.trace is not None
    assert result.trace.iterative_termination_reason == "error"


# ---------------------------------------------------------------------------
# Bonus: engine init validation rejects the bad config combination.
# ---------------------------------------------------------------------------


def test_engine_init_rejects_iterative_without_any_lm_client() -> None:
    """`enabled=True, decomposition_model=None, enrich_lm_client=None` -> ConfigurationError."""
    cfg = MagicMock(spec=RagServerConfig)
    cfg.persistence = SimpleNamespace(
        vector_store=None,
        document_store=MagicMock(),  # satisfy "at least one retrieval path"
        graph_store=None,
        metadata_store=None,
    )
    cfg.ingestion = SimpleNamespace(embeddings=None, sparse_embeddings=None, lm_client=None)
    cfg.retrieval = RetrievalConfig(
        top_k=5,
        adaptive=AdaptiveRetrievalConfig(enabled=False),
        iterative=IterativeRetrievalConfig(enabled=True, gate_mode="type"),
        enrich_lm_client=None,
    )
    cfg.tree_indexing = SimpleNamespace(enabled=False, model=None, max_tokens_per_node=10_000)
    cfg.tree_search = SimpleNamespace(enabled=False, model=None, max_context_tokens=200_000)
    cfg.routing = RoutingConfig(mode=QueryMode.RETRIEVAL)

    engine = RagEngine.__new__(RagEngine)
    engine._config = cfg

    with pytest.raises(ConfigurationError, match="iterative.decomposition_model"):
        engine._validate_config()
