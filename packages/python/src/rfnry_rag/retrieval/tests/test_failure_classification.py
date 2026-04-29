"""Heuristic failure classification on RetrievalTrace.

These tests exercise the pure-inspection classifier `classify_failure`
that converts a `RetrievalTrace` into a `FailureClassification` verdict.
The priority-order test (#8) is the regression guard: if a future
heuristic edit reorders evaluation, that test will catch it.

Bias-term hygiene: fixtures use generic identifiers (`q1`, `chunk_a`,
`R-101`, `EntityXYZ`) rather than domain vocabulary, mirroring the
agnostic-prompt convention enforced by `test_baml_prompt_domain_agnostic`.
"""

from rfnry_rag.retrieval.common.models import RetrievalTrace, RetrievedChunk
from rfnry_rag.retrieval.modules.evaluation import (
    FailureClassification,
    FailureType,
    classify_failure,
)


def _chunk(chunk_id: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=chunk_id, source_id="s1", content="text", score=score)


def test_classify_vocabulary_mismatch() -> None:
    trace = RetrievalTrace(
        query="q1",
        per_method_results={
            "document": [],
            "vector": [_chunk("chunk_a", 0.21)],
        },
    )

    result = classify_failure("q1", trace)

    assert isinstance(result, FailureClassification)
    assert result.type is FailureType.VOCABULARY_MISMATCH
    assert result.signals["vector_top_score"] == 0.21
    assert result.signals["document_results_count"] == 0


def test_classify_chunk_boundary() -> None:
    trace = RetrievalTrace(
        query="q1",
        per_method_results={"vector": [_chunk("chunk_a", 0.85)]},
        fused_results=[_chunk("chunk_a", 0.85)],
        final_results=[_chunk("chunk_a", 0.85)],
        grounding_decision="ungrounded",
    )

    result = classify_failure("q1", trace)

    assert result.type is FailureType.CHUNK_BOUNDARY
    assert result.signals["top_score"] == 0.85
    assert result.signals["grounding_decision"] == "ungrounded"


def test_classify_scope_miss() -> None:
    trace = RetrievalTrace(
        query="q1",
        per_method_results={"vector": [], "document": [], "graph": []},
        knowledge_id="kb-1",
    )

    result = classify_failure("q1", trace)

    assert result.type is FailureType.SCOPE_MISS
    assert result.signals["knowledge_id"] == "kb-1"
    assert result.signals["all_methods_empty"] is True


def test_classify_entity_not_indexed() -> None:
    trace = RetrievalTrace(
        query="What is R-101",
        per_method_results={
            "graph": [],
            "vector": [_chunk("chunk_a", 0.5)],
        },
    )

    result = classify_failure("What is R-101", trace)

    assert result.type is FailureType.ENTITY_NOT_INDEXED
    assert result.signals["matched_token"] == "R-101"
    assert result.signals["graph_results_count"] == 0


def test_classify_low_relevance() -> None:
    trace = RetrievalTrace(
        query="q1",
        per_method_results={
            "vector": [_chunk("chunk_a", 0.22)],
            "document": [_chunk("chunk_b", 0.18)],
        },
    )

    result = classify_failure("q1", trace)

    assert result.type is FailureType.LOW_RELEVANCE
    assert result.signals["max_score_observed"] == 0.22


def test_classify_insufficient_context() -> None:
    chunks = [_chunk(f"c{i}", 0.5) for i in range(3)]
    trace = RetrievalTrace(
        query="q1",
        per_method_results={"vector": chunks},
        fused_results=chunks,
        final_results=chunks,
        grounding_decision="ungrounded",
    )

    result = classify_failure("q1", trace)

    assert result.type is FailureType.INSUFFICIENT_CONTEXT
    assert result.signals["final_results_count"] == 3
    assert result.signals["grounding_decision"] == "ungrounded"


def test_classify_unknown_fallback() -> None:
    trace = RetrievalTrace(query="q1")

    result = classify_failure("q1", trace)

    assert result.type is FailureType.UNKNOWN


def test_classify_priority_order_vocabulary_beats_low_relevance() -> None:
    """VOCABULARY_MISMATCH must win when both heuristics fire.

    Document method empty + low vector top-score satisfies VOCABULARY_MISMATCH;
    the same chunk's score (0.2) is below `_LOW_RELEVANCE_THRESHOLD` (0.3),
    which would otherwise satisfy LOW_RELEVANCE. Priority order is the
    contract.
    """
    trace = RetrievalTrace(
        query="q1",
        per_method_results={
            "document": [],
            "vector": [_chunk("chunk_a", 0.2)],
        },
    )

    result = classify_failure("q1", trace)

    assert result.type is FailureType.VOCABULARY_MISMATCH
