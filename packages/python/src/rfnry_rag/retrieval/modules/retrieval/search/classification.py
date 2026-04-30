"""Query classification — heuristic + LLM.

Pure-async `classify_query(text, lm_client=None) -> QueryClassification`.
Heuristic path runs by default (free, deterministic, microsecond-scale
regex); LLM path opts in via a `LanguageModelClient` and pays one extra
LLM call (~$0.001, ~200 ms on Sonnet-class) for higher accuracy on
ambiguous queries.

The `QueryType` 4-label allowlist (FACTUAL / COMPARATIVE /
ENTITY_RELATIONSHIP / PROCEDURAL) describes query SHAPE — not domain.
Adding a fifth label requires explicit redesign across this module, the
BAML enum, and downstream consumers (task-aware weights, confidence
escalation, multi-hop iterative retrieval).

`_compute_adaptive_params` is the shared entry point that
`RetrievalService` (dynamic top_k + per-method weight multipliers) and
the confidence-expansion / multi-hop paths call. The confidence path
escalates on `complexity` confidence; the multi-hop path reuses the same
classifier verdict to decide iterative entry.
"""

from __future__ import annotations

import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Literal

from rfnry_rag.retrieval.baml.baml_client.async_client import b
from rfnry_rag.retrieval.common.language_model import LanguageModelClient, build_registry
from rfnry_rag.retrieval.common.logging import get_logger

if TYPE_CHECKING:
    from rfnry_rag.retrieval.server import AdaptiveRetrievalConfig

logger = get_logger("retrieval.search.classification")


# Generic "entity-like" token: leading capital, then 2+ chars from
# {A-Z, 0-9, _, -}. Matches `R-101`, `EntityXYZ`, `ServiceABC`. Excludes
# single-char tokens (`R`) and pure-lowercase words. Domain vocabulary
# is intentionally NOT baked in (Convention 1). This is the canonical
# entity-shape regex for the SDK; the failure analyser imports it from
# here so the two paths cannot drift.
_ENTITY_TOKEN_PATTERN = re.compile(r"\b[A-Z][A-Z0-9_-]{2,}\b")

# Module-level compiled regexes — compiled once at import, not per call.
_COMPARATIVE_PATTERN = re.compile(r"\b(compare|contrast|versus|vs\.?|differ|difference between)\b", re.IGNORECASE)
_PROCEDURAL_PATTERN = re.compile(r"\b(how (do|to|does)|steps to|procedure for|process for)\b", re.IGNORECASE)
_RELATIONAL_VERB_PATTERN = re.compile(r"\b(relate|connect|link|between|with)\b", re.IGNORECASE)


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
    consumers (adaptive parameters, confidence expansion) use these to
    tune top_k, method weights, and expansion thresholds without
    re-running the classifier.
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

    if length >= 200 or entity_count >= 3 or query_type in {QueryType.COMPARATIVE, QueryType.ENTITY_RELATIONSHIP}:
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
    # Mirrors HYBRID mode's "degrade to RAG on answerability check failure".
    # `build_registry` is inside the try because it can raise
    # `ConfigurationError` on `BOUNDARY_API_KEY` collision — covering it
    # keeps the "never raises" promise in `classify_query`'s docstring.
    try:
        registry = build_registry(lm_client)
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
        logger.warning("query classification LLM call failed; falling back to heuristic: %s", exc)
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


# Research-informed defaults — calibrated empirically against recall/F1
# deltas across query-type buckets. Keys are `QueryType.name` (uppercase);
# methods absent from a profile fall back to multiplier 1.0 in the lookup
# helper.
_DEFAULT_TASK_WEIGHT_PROFILES: dict[str, dict[str, float]] = {
    "FACTUAL": {"vector": 1.2, "document": 0.8, "graph": 0.8, "tree": 0.8},
    "COMPARATIVE": {"vector": 0.8, "document": 1.2, "graph": 0.8, "tree": 1.2},
    "ENTITY_RELATIONSHIP": {"vector": 0.8, "document": 0.8, "graph": 1.5, "tree": 0.8},
    "PROCEDURAL": {"vector": 1.0, "document": 1.2, "graph": 0.8, "tree": 1.2},
}


async def _compute_adaptive_params(
    query: str,
    base_top_k: int,
    config: AdaptiveRetrievalConfig,
    lm_client: LanguageModelClient | None,
    classify_fn: Callable[[str, LanguageModelClient | None], Awaitable[QueryClassification]],
) -> tuple[QueryClassification, int, dict[str, float], float]:
    """Run the classifier once and derive per-query top_k + method multipliers.

    `classify_fn` is parameterised so each consumer (currently `RetrievalService`)
    can supply its own module-local reference to `classify_query`. That keeps the
    test surface predictable: tests patch the consumer's import (e.g.
    `service.classify_query`) and the patch reaches every adaptive call without
    helper-internal binding shielding it.

    Returned tuple is `(classification, effective_top_k, multipliers, elapsed_seconds)`:
    - `classification` is exposed so the trace can record the verdict alongside the
      derived parameters without forcing the caller to re-await `classify_query`.
    - `effective_top_k` maps `QueryComplexity` -> int. MODERATE intentionally maps
      to `base_top_k` (the static `RetrievalConfig.top_k`) rather than to a
      separate config field — promoting MODERATE to its own knob would create a
      third tunable to calibrate against, with no observable behavioural gain
      over "the existing default".
    - `multipliers` maps method-name -> float using consumer-provided
      `task_weight_profiles` when present, else `_DEFAULT_TASK_WEIGHT_PROFILES`.
      Override semantics are per-QueryType-key fallback to defaults: a consumer
      who provides only the FACTUAL profile still gets the default profiles for
      the other three query types (so classifying as COMPARATIVE applies the
      default COMPARATIVE multipliers, not `{}`). Inside a profile, methods
      absent from the dict fall back to multiplier 1.0 (no change). This matches
      how consumers think about "I want a custom FACTUAL profile" without
      forcing them to re-state every method per-key, AND without silently
      stripping the other QueryType profiles.
    - `elapsed_seconds` is the wall-clock time spent in classification, surfaced
      so the trace can populate `timings["classification"]` without re-measuring.
    """
    start = time.perf_counter()
    classifier_lm = lm_client if config.use_llm_classification else None
    classification = await classify_fn(query, classifier_lm)
    elapsed = time.perf_counter() - start

    if classification.complexity is QueryComplexity.SIMPLE:
        effective_top_k = config.top_k_min
    elif classification.complexity is QueryComplexity.COMPLEX:
        effective_top_k = config.top_k_max
    else:
        effective_top_k = base_top_k

    # Per-QueryType-key fallback to defaults: a consumer who provides only
    # the FACTUAL profile gets defaults for the other three query types.
    # Full replacement at the dict level would mean classifying as
    # COMPARATIVE returns {} (no boost), contradicting the documented
    # contract — see the `multipliers` paragraph in this docstring.
    overrides = config.task_weight_profiles or {}
    profile = overrides.get(classification.query_type.name) or _DEFAULT_TASK_WEIGHT_PROFILES.get(
        classification.query_type.name, {}
    )
    multipliers = dict(profile)

    return classification, effective_top_k, multipliers, elapsed
