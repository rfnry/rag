"""Query classification (R5.1) — heuristic + LLM, plumbing for R5.2/R5.3/R6.

Pure-async `classify_query(text, lm_client=None) -> QueryClassification`.
Heuristic path runs by default (free, deterministic, microsecond-scale
regex); LLM path opts in via a `LanguageModelClient` and pays one extra
LLM call (~$0.001, ~200 ms on Sonnet-class) for higher accuracy on
ambiguous queries.

The `QueryType` 4-label allowlist (FACTUAL / COMPARATIVE /
ENTITY_RELATIONSHIP / PROCEDURAL) describes query SHAPE — not domain.
Adding a fifth label requires explicit redesign across this module, the
BAML enum, and downstream consumers (R5.2 task-aware weights, R5.3
confidence escalation, R6 multi-hop).

R5.1 is invisible plumbing: nothing in `RetrievalService.retrieve`
consumes the classifier yet. R5.2 starts dispatch on `query_type`; R5.3
escalates on `complexity` confidence; R6 will share the same classifier
to decide multi-hop entry.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from rfnry_rag.retrieval.baml.baml_client.async_client import b
from rfnry_rag.retrieval.common.language_model import LanguageModelClient, build_registry
from rfnry_rag.retrieval.common.logging import get_logger

logger = get_logger("retrieval.search.classification")


# Generic "entity-like" token: leading capital, then 2+ chars from
# {A-Z, 0-9, _, -}. Matches `R-101`, `EntityXYZ`, `ServiceABC`. Excludes
# single-char tokens (`R`) and pure-lowercase words. Domain vocabulary
# is intentionally NOT baked in (Convention 1). This is the canonical
# entity-shape regex for the SDK; R8.2's failure analyser imports it
# from here so the two paths cannot drift.
_ENTITY_TOKEN_PATTERN = re.compile(r"\b[A-Z][A-Z0-9_-]{2,}\b")

# Module-level compiled regexes — compiled once at import, not per call.
_COMPARATIVE_PATTERN = re.compile(
    r"\b(compare|contrast|versus|vs\.?|differ|difference between)\b", re.IGNORECASE
)
_PROCEDURAL_PATTERN = re.compile(
    r"\b(how (do|to|does)|steps to|procedure for|process for)\b", re.IGNORECASE
)
_RELATIONAL_VERB_PATTERN = re.compile(
    r"\b(relate|connect|link|between|with)\b", re.IGNORECASE
)


class QueryComplexity(Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


class QueryType(Enum):
    FACTUAL = "factual"
    COMPARATIVE = "comparative"
    ENTITY_RELATIONSHIP = "entity_relationship"
    PROCEDURAL = "procedural"


@dataclass(frozen=True)
class QueryClassification:
    """Verdict from `classify_query`.

    `signals` carries the indicators that drove the verdict — the
    heuristic path reports `query_length`, `entity_count`, and a capped
    list of `entities`; the LLM path reports `llm_reasoning`. Downstream
    consumers (R5.2/R5.3) use these to tune top_k, method weights, and
    expansion thresholds without re-running the classifier.
    """

    complexity: QueryComplexity
    query_type: QueryType
    signals: dict[str, str | int | float | bool | None | list[str]] = field(default_factory=dict)
    source: Literal["heuristic", "llm"] = "heuristic"


def _heuristic_classify(text: str) -> QueryClassification:
    entities = _ENTITY_TOKEN_PATTERN.findall(text)
    entity_count = len(entities)
    length = len(text)

    # Priority order: ENTITY_RELATIONSHIP > COMPARATIVE > PROCEDURAL > FACTUAL.
    # ENTITY_RELATIONSHIP wins over PROCEDURAL when "how does R-101 connect
    # to V-203?" matches both, because the entity-pair signal is more
    # specific than the procedural verb form.
    if entity_count >= 2 and _RELATIONAL_VERB_PATTERN.search(text):
        query_type = QueryType.ENTITY_RELATIONSHIP
    elif _COMPARATIVE_PATTERN.search(text):
        query_type = QueryType.COMPARATIVE
    elif _PROCEDURAL_PATTERN.search(text):
        query_type = QueryType.PROCEDURAL
    else:
        query_type = QueryType.FACTUAL

    if (
        length >= 200
        or entity_count >= 3
        or query_type in {QueryType.COMPARATIVE, QueryType.ENTITY_RELATIONSHIP}
    ):
        complexity = QueryComplexity.COMPLEX
    elif length <= 60 and entity_count <= 1 and query_type is QueryType.FACTUAL:
        complexity = QueryComplexity.SIMPLE
    else:
        complexity = QueryComplexity.MODERATE

    signals: dict[str, str | int | float | bool | None | list[str]] = {
        "query_length": length,
        "entity_count": entity_count,
        "entities": entities[:5],
    }
    return QueryClassification(
        complexity=complexity,
        query_type=query_type,
        signals=signals,
        source="heuristic",
    )


async def _llm_classify(text: str, lm_client: LanguageModelClient) -> QueryClassification:
    # Fall back to the heuristic on any LLM failure: a classifier failure
    # must never make the SDK worse than running with no classifier.
    # Mirrors R1.3's "degrade to RAG on answerability check failure".
    registry = build_registry(lm_client)
    try:
        verdict = await b.ClassifyQueryComplexity(
            query=text,
            baml_options={"client_registry": registry},
        )
        return QueryClassification(
            complexity=QueryComplexity[verdict.complexity.value.upper()],
            query_type=QueryType[verdict.query_type.value.upper()],
            signals={"llm_reasoning": verdict.reasoning},
            source="llm",
        )
    except Exception as exc:
        logger.warning(
            "query classification LLM call failed; falling back to heuristic: %s", exc
        )
        return _heuristic_classify(text)


async def classify_query(
    text: str,
    lm_client: LanguageModelClient | None = None,
) -> QueryClassification:
    """Classify a query into (complexity, query_type).

    With `lm_client=None` (default): pure-regex heuristic; deterministic,
    microsecond-scale, no I/O. With `lm_client` provided: BAML-backed LLM
    call, ~200 ms + ~$0.001 per query, more accurate on ambiguous text.

    Never raises — on LLM exception, logs a warning and returns the
    heuristic result so a classifier failure cannot block retrieval.
    The function is `async` even on the heuristic path so callers don't
    need to branch on which path was taken.
    """
    if lm_client is None:
        return _heuristic_classify(text)
    return await _llm_classify(text, lm_client)
