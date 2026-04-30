"""Dynamic top_k + task-aware method weights.

These tests cover per-query `top_k` adjustment based on classified
complexity, and per-query method-weight multipliers based on classified
query type. Both consume `classify_query` once per
`RetrievalService.retrieve` call (BEFORE query rewriting, so the
classifier sees the original user query rather than LLM-generated
variants).

When `AdaptiveRetrievalConfig.enabled=False` (default), the pipeline runs
byte-for-byte unchanged — no classifier call, no multipliers, `trace.adaptive`
stays None.

Bias-term hygiene: fixtures use neutral identifiers (`q1`, `chunk_a`, `R-101`).
No domain-specific vocabulary.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from rfnry_rag.retrieval.common.models import RetrievedChunk
from rfnry_rag.retrieval.modules.retrieval.search.classification import (
    QueryClassification,
    QueryComplexity,
    QueryType,
)
from rfnry_rag.retrieval.modules.retrieval.search.service import RetrievalService
from rfnry_rag.retrieval.server import AdaptiveRetrievalConfig


def _chunk(chunk_id: str, score: float = 0.9) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=chunk_id, source_id="s1", content="text", score=score)


def _mock_method(name: str, results: list[RetrievedChunk], weight: float = 1.0) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        weight=weight,
        top_k=None,
        search=AsyncMock(return_value=results),
    )


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


_CLASSIFY_PATH = "rfnry_rag.retrieval.modules.retrieval.search.service.classify_query"


async def test_adaptive_disabled_runs_existing_pipeline_byte_for_byte() -> None:
    """`adaptive=None` (or `enabled=False`) must skip the classifier entirely."""
    method_a = _mock_method("vector", [_chunk("chunk_a")])
    service = RetrievalService(retrieval_methods=[method_a], top_k=5)

    with patch(_CLASSIFY_PATH, new=AsyncMock()) as mock_classify:
        chunks, trace = await service.retrieve(query="q1", trace=True)

    mock_classify.assert_not_awaited()
    assert trace is not None
    assert trace.adaptive is None
    assert chunks


async def test_adaptive_dynamic_topk_simple_uses_top_k_min() -> None:
    """SIMPLE complexity -> effective top_k = `top_k_min` (default 3)."""
    method_a = _mock_method("vector", [_chunk("chunk_a")])
    adaptive = AdaptiveRetrievalConfig(enabled=True, top_k_min=3, top_k_max=15)
    service = RetrievalService(retrieval_methods=[method_a], top_k=5, adaptive_config=adaptive)

    with patch(
        _CLASSIFY_PATH,
        new=AsyncMock(return_value=_classification(complexity=QueryComplexity.SIMPLE)),
    ):
        _chunks, trace = await service.retrieve(query="q1", trace=True)

    assert trace is not None
    assert trace.adaptive is not None
    assert trace.adaptive["effective_top_k"] == 3


async def test_adaptive_dynamic_topk_moderate_uses_base_top_k() -> None:
    """MODERATE complexity -> effective top_k = base `RetrievalConfig.top_k` (5)."""
    method_a = _mock_method("vector", [_chunk("chunk_a")])
    adaptive = AdaptiveRetrievalConfig(enabled=True, top_k_min=3, top_k_max=15)
    service = RetrievalService(retrieval_methods=[method_a], top_k=5, adaptive_config=adaptive)

    with patch(
        _CLASSIFY_PATH,
        new=AsyncMock(return_value=_classification(complexity=QueryComplexity.MODERATE)),
    ):
        _chunks, trace = await service.retrieve(query="q1", trace=True)

    assert trace is not None
    assert trace.adaptive is not None
    assert trace.adaptive["effective_top_k"] == 5


async def test_adaptive_dynamic_topk_complex_uses_top_k_max() -> None:
    """COMPLEX complexity -> effective top_k = `top_k_max` (15)."""
    method_a = _mock_method("vector", [_chunk("chunk_a")])
    adaptive = AdaptiveRetrievalConfig(enabled=True, top_k_min=3, top_k_max=15)
    service = RetrievalService(retrieval_methods=[method_a], top_k=5, adaptive_config=adaptive)

    with patch(
        _CLASSIFY_PATH,
        new=AsyncMock(return_value=_classification(complexity=QueryComplexity.COMPLEX)),
    ):
        _chunks, trace = await service.retrieve(query="q1", trace=True)

    assert trace is not None
    assert trace.adaptive is not None
    assert trace.adaptive["effective_top_k"] == 15


async def test_adaptive_task_weights_entity_relationship_boosts_graph_method() -> None:
    """ENTITY_RELATIONSHIP profile multiplies graph weight by 1.5, others by 0.8."""
    method_v = _mock_method("vector", [_chunk("chunk_v")])
    method_d = _mock_method("document", [_chunk("chunk_d")])
    method_g = _mock_method("graph", [_chunk("chunk_g")])
    adaptive = AdaptiveRetrievalConfig(enabled=True)
    service = RetrievalService(retrieval_methods=[method_v, method_d, method_g], top_k=5, adaptive_config=adaptive)

    with patch(
        _CLASSIFY_PATH,
        new=AsyncMock(return_value=_classification(query_type=QueryType.ENTITY_RELATIONSHIP)),
    ):
        _chunks, trace = await service.retrieve(query="q1", trace=True)

    assert trace is not None
    assert trace.adaptive is not None
    multipliers = trace.adaptive["applied_multipliers"]
    assert multipliers["vector"] == 0.8
    assert multipliers["document"] == 0.8
    assert multipliers["graph"] == 1.5


async def test_adaptive_task_weights_factual_boosts_vector_dominant() -> None:
    """FACTUAL profile multiplies vector by 1.2, others by 0.8."""
    method_v = _mock_method("vector", [_chunk("chunk_v")])
    method_d = _mock_method("document", [_chunk("chunk_d")])
    method_g = _mock_method("graph", [_chunk("chunk_g")])
    adaptive = AdaptiveRetrievalConfig(enabled=True)
    service = RetrievalService(retrieval_methods=[method_v, method_d, method_g], top_k=5, adaptive_config=adaptive)

    with patch(
        _CLASSIFY_PATH,
        new=AsyncMock(return_value=_classification(query_type=QueryType.FACTUAL)),
    ):
        _chunks, trace = await service.retrieve(query="q1", trace=True)

    assert trace is not None
    assert trace.adaptive is not None
    multipliers = trace.adaptive["applied_multipliers"]
    assert multipliers["vector"] == 1.2
    assert multipliers["document"] == 0.8
    assert multipliers["graph"] == 0.8


async def test_adaptive_partial_override_falls_back_to_defaults_for_other_query_types() -> None:
    """Consumer provides FACTUAL only — COMPARATIVE classification must fall
    back to the default COMPARATIVE profile, not return `{}`.

    Regression guard for the documented partial-override contract: full
    replacement at the dict level would mean classifying as a query type
    the consumer didn't override gives no boost — contradicting plan,
    docstring, and CHANGELOG.
    """
    method_v = _mock_method("vector", [_chunk("chunk_v")])
    method_d = _mock_method("document", [_chunk("chunk_d")])
    method_g = _mock_method("graph", [_chunk("chunk_g")])
    # Distinguishable override: vector=9.0 is wildly different from any default,
    # so we can tell unambiguously whether the consumer's profile or the
    # default profile is in effect.
    adaptive = AdaptiveRetrievalConfig(
        enabled=True,
        task_weight_profiles={"FACTUAL": {"vector": 9.0}},
    )
    service = RetrievalService(retrieval_methods=[method_v, method_d, method_g], top_k=5, adaptive_config=adaptive)

    # Classify as COMPARATIVE — the consumer DID NOT override this profile,
    # so the default COMPARATIVE multipliers must be applied.
    with patch(
        _CLASSIFY_PATH,
        new=AsyncMock(return_value=_classification(query_type=QueryType.COMPARATIVE)),
    ):
        _chunks, trace = await service.retrieve(query="q1", trace=True)

    assert trace is not None
    assert trace.adaptive is not None
    multipliers = trace.adaptive["applied_multipliers"]
    # Default COMPARATIVE profile: vector=0.8, document=1.2, graph=0.8, tree=1.2
    assert multipliers["vector"] == 0.8
    assert multipliers["document"] == 1.2
    assert multipliers["graph"] == 0.8
    assert multipliers["tree"] == 1.2

    # Sibling assertion: classifying as FACTUAL applies the consumer's override,
    # confirming the override path still works (vector=9.0, not the default 1.2).
    method_v.search.reset_mock()
    method_d.search.reset_mock()
    method_g.search.reset_mock()
    with patch(
        _CLASSIFY_PATH,
        new=AsyncMock(return_value=_classification(query_type=QueryType.FACTUAL)),
    ):
        _chunks, trace = await service.retrieve(query="q1", trace=True)

    assert trace is not None
    assert trace.adaptive is not None
    assert trace.adaptive["applied_multipliers"]["vector"] == 9.0


async def test_adaptive_tree_multiplier_applied_when_tree_chunks_present() -> None:
    """Tree search merges into fusion at a separate site — multiplier must apply.

    Regression guard for the tree-multiplier-unreachable bug: tree_chunks
    are appended to the fusion pool with a hardcoded weight, bypassing
    `_search_single_query`'s per-method multiplier application. The
    default profiles' `tree` entries (1.2 / 0.8) must affect the weight
    pushed into RRF — otherwise operators reading
    `trace.adaptive["applied_multipliers"]["tree"]` would be misled.
    """
    method_v = _mock_method("vector", [_chunk("chunk_v")])
    adaptive = AdaptiveRetrievalConfig(enabled=True)
    service = RetrievalService(retrieval_methods=[method_v], top_k=5, adaptive_config=adaptive)

    # COMPARATIVE has tree=1.2 in the default profile.
    classify_patch = patch(
        _CLASSIFY_PATH,
        new=AsyncMock(return_value=_classification(query_type=QueryType.COMPARATIVE)),
    )
    rrf_patch = patch(
        "rfnry_rag.retrieval.modules.retrieval.search.service.reciprocal_rank_fusion",
        return_value=[],
    )
    with classify_patch, rrf_patch as mock_rrf:
        _chunks, trace = await service.retrieve(
            query="q1",
            tree_chunks=[_chunk("chunk_tree")],
            trace=True,
        )

    assert trace is not None
    assert trace.adaptive is not None
    assert trace.adaptive["applied_multipliers"]["tree"] == 1.2

    # Spy on the fusion call: the weights array passed to RRF must include
    # 1.2 from the tree path (1.0 hardcoded * 1.2 multiplier).
    assert mock_rrf.called
    _args, kwargs = mock_rrf.call_args
    method_weights = kwargs["method_weights"]
    assert 1.2 in method_weights


async def test_adaptive_trace_records_classification_and_effective_topk() -> None:
    """`trace.adaptive` carries complexity, query_type, top_k, multipliers, source."""
    method_v = _mock_method("vector", [_chunk("chunk_v")])
    adaptive = AdaptiveRetrievalConfig(enabled=True, top_k_min=3, top_k_max=15)
    service = RetrievalService(retrieval_methods=[method_v], top_k=5, adaptive_config=adaptive)

    with patch(
        _CLASSIFY_PATH,
        new=AsyncMock(return_value=_classification(complexity=QueryComplexity.MODERATE, query_type=QueryType.FACTUAL)),
    ):
        _chunks, trace = await service.retrieve(query="q1", trace=True)

    assert trace is not None
    assert trace.adaptive is not None
    assert trace.adaptive["complexity"] == "MODERATE"
    assert trace.adaptive["query_type"] == "FACTUAL"
    assert trace.adaptive["effective_top_k"] == 5
    assert trace.adaptive["applied_multipliers"]["vector"] == 1.2
    assert trace.adaptive["classification_source"] == "heuristic"
    assert "classification" in trace.timings
