"""Heuristic failure classification on `RetrievalTrace` (R8.2).

Pure-inspection function that turns a `RetrievalTrace` (from R8.1) into a
small `FailureType` enum verdict plus the trace-derived `signals` that
drove the decision. No LLM calls, no I/O, no new dependencies — the
classifier is the cheap first pass that lets a benchmark report tell the
user which class of failure dominates their workload (e.g., "40% of
failures are vocabulary_mismatch — enable R3 expansion to fix the
dominant class") without paying per-case LLM cost.

Caller's responsibility to invoke only on failed cases. The classifier
does not itself decide pass/fail — R8.3's benchmark harness will.

Threshold rationale
-------------------
The three module-private constants below are heuristic defaults picked
by intuition, not benchmark — R8.3's benchmark provides the data needed
to tune them later. They are NOT consumer-facing config: they live at
module scope, NOT on a config dataclass, and they do NOT enter
`_CONFIGS_TO_AUDIT`. If a real consumer needs to tune them, promote to
a `FailureAnalysisConfig` then.

The `signals` dict on every classification reports the trace-derived
values that drove the decision (including the threshold compared
against), so a downstream consumer can override on their own corpus
without us having to expose the knob.

Priority (first-match wins)
---------------------------
1. `VOCABULARY_MISMATCH`    — keyword-channel produced nothing AND vector
                              top score < `_VOCABULARY_MISMATCH_THRESHOLD`
2. `CHUNK_BOUNDARY`         — high-score top result AND grounding ungrounded
3. `SCOPE_MISS`             — every method empty AND `knowledge_id` set
4. `ENTITY_NOT_INDEXED`     — graph empty AND query has entity-like token
5. `LOW_RELEVANCE`          — methods returned results, max score under floor
6. `INSUFFICIENT_CONTEXT`   — final results non-empty but grounding ungrounded
7. `UNKNOWN`                — fallback when no heuristic matches

VOCABULARY_MISMATCH is checked before LOW_RELEVANCE because the
"document method empty" signal is more specific than "max score
low" — the same chunk scoring 0.2 could be diagnosed either way, and
the more specific cause wins. CHUNK_BOUNDARY beats INSUFFICIENT_CONTEXT
by the same logic: a high-score top result narrows the cause to a
boundary issue, where INSUFFICIENT_CONTEXT just says "chunks were
relevant but didn't suffice".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from rfnry_rag.retrieval.common.logging import get_logger
from rfnry_rag.retrieval.common.models import RetrievalTrace, RetrievedChunk

logger = get_logger("evaluation.failure_analysis")


# Empirical defaults — see "Threshold rationale" in the module docstring.
# A vector top-score below 0.4 with the document/BM25 method coming back
# empty is a strong vocabulary-mismatch signal (cosine ~0.4 is roughly
# "loosely related" for typical embedding models).
_VOCABULARY_MISMATCH_THRESHOLD = 0.4
# A top-score >= 0.7 means the retrieval channel found a clearly relevant
# chunk; if grounding still failed, the answer likely spans an adjacent
# chunk that didn't rank.
_HIGH_RELEVANCE_THRESHOLD = 0.7
# No chunk above 0.3 across any method is the floor below which results
# are noise even if non-empty.
_LOW_RELEVANCE_THRESHOLD = 0.3

# Generic "entity-like" token: leading capital, then 2+ chars from
# {A-Z, 0-9, _, -}. Matches `R-101`, `PumpModelX`, `EntityXYZ`,
# `ServiceABC`. Excludes single-char tokens (`R`) and pure-lowercase
# words. Domain vocabulary is intentionally NOT baked in (Convention 1).
# The regex is deliberately permissive — a `JSON` or `HTTP` will match
# too. The `signals["matched_token"]` field reports what triggered the
# classification so consumers can judge.
_ENTITY_TOKEN_PATTERN = re.compile(r"\b[A-Z][A-Z0-9_-]{2,}\b")


class FailureType(Enum):
    VOCABULARY_MISMATCH = "vocabulary_mismatch"
    CHUNK_BOUNDARY = "chunk_boundary"
    SCOPE_MISS = "scope_miss"
    ENTITY_NOT_INDEXED = "entity_not_indexed"
    LOW_RELEVANCE = "low_relevance"
    INSUFFICIENT_CONTEXT = "insufficient_context"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class FailureClassification:
    """Verdict from `classify_failure`.

    `signals` carries the trace-derived values that drove the
    classification (including the threshold compared against), making
    the heuristic explainable AND letting a downstream consumer override
    thresholds based on their corpus.
    """

    type: FailureType
    reasoning: str
    signals: dict[str, str | float | int | bool | None] = field(default_factory=dict)


def _max_score(chunks: list[RetrievedChunk]) -> float | None:
    if not chunks:
        return None
    return max(c.score for c in chunks)


def _top_score_from_trace(trace: RetrievalTrace) -> float | None:
    # CHUNK_BOUNDARY's "top result" can come from final_results when it's
    # populated, falling back to fused_results — final is post-rerank /
    # post-refine, so it's the more authoritative ranking when present.
    # Explicit `is not None` guard rather than truthy fallback so a
    # legitimate 0.0 score doesn't silently route to fused_results.
    if trace.final_results:
        return _max_score(trace.final_results)
    return _max_score(trace.fused_results)


def classify_failure(query: str, trace: RetrievalTrace) -> FailureClassification:
    """Inspect a failed-query trace and return the most likely failure category.

    Order of evaluation matters — first-match wins. See module docstring
    for the rationale behind each heuristic and the priority order.
    Caller is responsible for invoking only on cases that actually
    failed; this function does not itself decide pass/fail.
    """
    per_method = trace.per_method_results

    document_results = per_method.get("document", [])
    vector_results = per_method.get("vector", [])
    graph_results = per_method.get("graph", [])

    vector_top = _max_score(vector_results)
    if not document_results and vector_top is not None and vector_top < _VOCABULARY_MISMATCH_THRESHOLD:
        signals: dict[str, str | float | int | bool | None] = {
            "document_results_count": 0,
            "vector_top_score": vector_top,
            "vocabulary_mismatch_threshold": _VOCABULARY_MISMATCH_THRESHOLD,
        }
        reasoning = (
            f"Document/keyword channel returned no results and the vector top score "
            f"{vector_top:.3f} is below {_VOCABULARY_MISMATCH_THRESHOLD} — likely a "
            f"vocabulary gap between query and corpus."
        )
        logger.debug("failure=vocabulary_mismatch %s", signals)
        return FailureClassification(
            type=FailureType.VOCABULARY_MISMATCH, reasoning=reasoning, signals=signals
        )

    top_score = _top_score_from_trace(trace)
    if (
        top_score is not None
        and top_score >= _HIGH_RELEVANCE_THRESHOLD
        and trace.grounding_decision == "ungrounded"
    ):
        signals = {
            "top_score": top_score,
            "high_relevance_threshold": _HIGH_RELEVANCE_THRESHOLD,
            "grounding_decision": trace.grounding_decision,
        }
        reasoning = (
            f"Top retrieved chunk scored {top_score:.3f} (>= {_HIGH_RELEVANCE_THRESHOLD}) "
            f"but grounding still failed — the answer likely spans the high-score chunk "
            f"and an adjacent one that did not rank."
        )
        logger.debug("failure=chunk_boundary %s", signals)
        return FailureClassification(
            type=FailureType.CHUNK_BOUNDARY, reasoning=reasoning, signals=signals
        )

    # SCOPE_MISS requires per_method to be populated AND every list empty.
    # An entirely-absent per_method dict (no methods configured) falls
    # through to UNKNOWN — that's a config issue, not a scope issue.
    if per_method and all(len(v) == 0 for v in per_method.values()) and trace.knowledge_id is not None:
        signals = {
            "knowledge_id": trace.knowledge_id,
            "all_methods_empty": True,
        }
        reasoning = (
            f"Every retrieval method returned [] within knowledge_id={trace.knowledge_id!r} "
            f"— either the wrong scope or no relevant content in this scope."
        )
        logger.debug("failure=scope_miss %s", signals)
        return FailureClassification(
            type=FailureType.SCOPE_MISS, reasoning=reasoning, signals=signals
        )

    if "graph" in per_method and not graph_results:
        match = _ENTITY_TOKEN_PATTERN.search(query)
        if match is not None:
            signals = {
                "graph_results_count": 0,
                "matched_token": match.group(0),
            }
            reasoning = (
                f"Query references entity-like token {match.group(0)!r} but the graph "
                f"channel returned [] — entity may not be indexed."
            )
            logger.debug("failure=entity_not_indexed %s", signals)
            return FailureClassification(
                type=FailureType.ENTITY_NOT_INDEXED, reasoning=reasoning, signals=signals
            )

    max_observed: float | None = None
    for chunks in per_method.values():
        method_max = _max_score(chunks)
        if method_max is not None and (max_observed is None or method_max > max_observed):
            max_observed = method_max
    any_method_non_empty = any(len(v) > 0 for v in per_method.values())
    if any_method_non_empty and max_observed is not None and max_observed <= _LOW_RELEVANCE_THRESHOLD:
        signals = {
            "max_score_observed": max_observed,
            "low_relevance_threshold": _LOW_RELEVANCE_THRESHOLD,
        }
        reasoning = (
            f"Methods returned results but the maximum score {max_observed:.3f} is at or "
            f"below {_LOW_RELEVANCE_THRESHOLD} — retrieved content is likely noise."
        )
        logger.debug("failure=low_relevance %s", signals)
        return FailureClassification(
            type=FailureType.LOW_RELEVANCE, reasoning=reasoning, signals=signals
        )

    if trace.final_results and trace.grounding_decision == "ungrounded":
        signals = {
            "final_results_count": len(trace.final_results),
            "grounding_decision": trace.grounding_decision,
        }
        reasoning = (
            f"{len(trace.final_results)} chunks survived the relevance gate but grounding "
            f"still failed — chunks were on-topic but did not carry enough information."
        )
        logger.debug("failure=insufficient_context %s", signals)
        return FailureClassification(
            type=FailureType.INSUFFICIENT_CONTEXT, reasoning=reasoning, signals=signals
        )

    signals = {"reason": "no heuristic matched"}
    logger.debug("failure=unknown %s", signals)
    return FailureClassification(
        type=FailureType.UNKNOWN,
        reasoning="No heuristic matched — trace shape does not fit any known failure mode.",
        signals=signals,
    )
