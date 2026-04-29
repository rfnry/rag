"""IterativeRetrievalService — sibling to RetrievalService for multi-hop queries.

R6.1 shipped the empty stub; R6.2 lands the runtime hop loop, decomposer
wiring, and trace population. R6.3 will add post-loop DIRECT escalation
on top of this service (handled at the engine layer, not here — service
concerns shouldn't know about cross-strategy escalation).

The loop is sequential by design (each hop depends on the prior hop's
findings) so there is no ``asyncio.gather`` over hops. Within a hop, the
existing ``RetrievalService`` parallelism (multi-method dispatch, query
rewriting) is unchanged.
"""

from __future__ import annotations

import time

from rfnry_rag.retrieval.baml.baml_client.async_client import b
from rfnry_rag.retrieval.baml.baml_client.types import DecomposeResult
from rfnry_rag.retrieval.common.language_model import LanguageModelClient, build_registry
from rfnry_rag.retrieval.common.logging import get_logger
from rfnry_rag.retrieval.common.models import RetrievedChunk
from rfnry_rag.retrieval.modules.retrieval.iterative.config import IterativeRetrievalConfig
from rfnry_rag.retrieval.modules.retrieval.iterative.trace import (
    IterativeHopTrace,
    IterativeOutcome,
)
from rfnry_rag.retrieval.modules.retrieval.search.classification import (
    QueryClassification,
    QueryComplexity,
    QueryType,
)
from rfnry_rag.retrieval.modules.retrieval.search.service import RetrievalService

logger = get_logger("retrieval.iterative.service")


def gate_passes_type(classification: QueryClassification | None) -> bool:
    """Type-mode gate: pass on COMPLEX or ENTITY_RELATIONSHIP queries.

    Conservative by design — single-fact / SIMPLE / FACTUAL queries do
    not benefit from multi-hop decomposition and would just burn
    decompose calls. The gate is bypassed when ``gate_mode="llm"`` (the
    decomposer becomes the gate via its first ``done=true`` call).
    """
    if classification is None:
        return False
    if classification.complexity is QueryComplexity.COMPLEX:
        return True
    return classification.query_type is QueryType.ENTITY_RELATIONSHIP


def _merge_chunks_dedup(
    accumulated: list[RetrievedChunk],
    new: list[RetrievedChunk],
) -> list[RetrievedChunk]:
    """Dedup by ``chunk_id``; on collision keep the higher-scored version.

    Insertion order is preserved for determinism. New chunks append in
    the order returned by the per-hop retrieve call. Collisions update
    the existing slot in place (do NOT move to end) — moving would
    destabilise iteration order across hops, which is observable in
    traces and tests.
    """
    if not accumulated:
        # Cheap fast path; also skips the index-build cost for hop 0.
        return list(new)
    by_id: dict[str, int] = {c.chunk_id: i for i, c in enumerate(accumulated)}
    out = list(accumulated)
    for chunk in new:
        if chunk.chunk_id in by_id:
            existing_idx = by_id[chunk.chunk_id]
            existing = out[existing_idx]
            if chunk.score > existing.score:
                out[existing_idx] = chunk
        else:
            by_id[chunk.chunk_id] = len(out)
            out.append(chunk)
    return out


class IterativeRetrievalService:
    """Multi-hop iterative retrieval over a wrapped ``RetrievalService``.

    The service owns the hop loop and the decomposer call; per-hop
    retrieval delegates to ``RetrievalService.retrieve`` unchanged so
    R5.2's adaptive classifier + R5.3's confidence expansion (when
    enabled on the underlying retrieval config) compose naturally with
    iterative without any iterative-specific plumbing.

    R5.3's LC escalation lives on ``RagEngine._query_via_retrieval``,
    NOT on ``RetrievalService.retrieve``. Per-hop calls go through the
    service directly so they naturally skip that escalation tail —
    iterative-then-DIRECT escalation is R6.3's job, layered on top of
    this service at the engine arm.
    """

    def __init__(
        self,
        retrieval_service: RetrievalService,
        fallback_decomposition_lm: LanguageModelClient | None = None,
    ) -> None:
        self._retrieval_service = retrieval_service
        # `fallback_decomposition_lm` lets the engine pass
        # `RetrievalConfig.enrich_lm_client` as the default decomposer when
        # `IterativeRetrievalConfig.decomposition_model` is unset. R5 uses
        # the same default-fallback pattern (`AdaptiveRetrievalConfig.
        # use_llm_classification` falls back to `enrich_lm_client`).
        self._fallback_decomposition_lm = fallback_decomposition_lm

    async def _call_decompose(
        self,
        original_query: str,
        accumulated_findings: str,
        hop_index: int,
        max_hops: int,
        decomposition_model: LanguageModelClient | None,
    ) -> DecomposeResult:
        """Run ``b.DecomposeQuery`` with a per-call BAML registry.

        Mirrors ``classify_query``'s LLM path: build a fresh registry
        scoped to the chosen client (``decomposition_model`` first; else
        ``fallback_decomposition_lm``). When neither is set the function
        raises — the engine init should have caught that earlier.
        """
        client = decomposition_model or self._fallback_decomposition_lm
        if client is None:
            raise RuntimeError(
                "IterativeRetrievalService.retrieve called without a decomposition "
                "client; either IterativeRetrievalConfig.decomposition_model or "
                "RetrievalConfig.enrich_lm_client must be set."
            )
        registry = build_registry(client)
        return await b.DecomposeQuery(
            original_query=original_query,
            accumulated_findings=accumulated_findings,
            hop_index=hop_index,
            max_hops=max_hops,
            baml_options={"client_registry": registry},
        )

    async def retrieve(
        self,
        query: str,
        knowledge_id: str | None,
        history: list[tuple[str, str]] | None,  # noqa: ARG002 - signature parity (see below)
        min_score: float | None,
        collection: str | None,  # noqa: ARG002 - signature parity (see below)
        *,
        trace: bool,  # noqa: ARG002 - hop-level trace is unconditional (see below)
        iterative: IterativeRetrievalConfig,
        classification: QueryClassification | None = None,
    ) -> tuple[list[RetrievedChunk], IterativeOutcome]:
        """Run the hop loop and return ``(accumulated_chunks, outcome)``.

        Sub-question retrieval calls ``RetrievalService.retrieve`` with
        ``trace=True`` regardless of the outer ``trace`` flag — the per-hop
        adaptive verdict, per-method results, and timings are needed to
        populate ``IterativeHopTrace``. The outer caller decides whether to
        surface the full hop trace via ``RetrievalTrace.iterative_hops``.

        ``classification`` is the pre-computed verdict from the engine arm
        (R5's classifier already runs there before deciding which
        ``_query_via_*`` to invoke). The service does not re-classify; in
        ``gate_mode="llm"`` the decomposer is the gate and the type check
        is bypassed entirely.

        ``history`` and ``collection`` are accepted for API parity with the
        other ``_query_via_*`` paths so the engine arm can call us
        symmetrically. ``history`` is intentionally NOT propagated per-hop
        — the decomposer's ``accumulated_findings`` carries the
        loop-internal state. ``collection`` is honoured at the engine layer
        when constructing the wrapped ``RetrievalService`` (each collection
        gets its own pipeline).
        """
        # Type-mode gate: short-circuit when classifier verdict is neither
        # COMPLEX nor ENTITY_RELATIONSHIP. The engine arm should have made
        # this same decision before delegating, but defending it here keeps
        # the service usable in isolation (e.g. for tests).
        if iterative.gate_mode == "type" and not gate_passes_type(classification):
            return [], IterativeOutcome(
                hops=[],
                termination_reason="done",
                total_decompose_calls=0,
                total_retrieve_calls=0,
            )

        accumulated_chunks: list[RetrievedChunk] = []
        accumulated_findings: str = ""
        hop_traces: list[IterativeHopTrace] = []
        decompose_calls = 0
        retrieve_calls = 0

        for hop_index in range(iterative.max_hops):
            decompose_calls += 1
            decompose_t0 = time.perf_counter()
            try:
                result = await self._call_decompose(
                    original_query=query,
                    accumulated_findings=accumulated_findings,
                    hop_index=hop_index,
                    max_hops=iterative.max_hops,
                    decomposition_model=iterative.decomposition_model,
                )
            except Exception as exc:
                # A decomposer failure mid-loop must not raise into the
                # engine — terminate cleanly so the caller can fall back
                # to plain retrieval with whatever chunks accumulated so far.
                logger.warning(
                    "decompose call failed at hop %d (%s); terminating loop with error",
                    hop_index,
                    exc,
                )
                hop_traces.append(
                    IterativeHopTrace(
                        hop_index=hop_index,
                        sub_question=None,
                        findings_from_last_hop=accumulated_findings,
                        decompose_reasoning=f"decompose failed: {exc}",
                        timings={"decompose": time.perf_counter() - decompose_t0},
                    )
                )
                return accumulated_chunks, IterativeOutcome(
                    hops=hop_traces,
                    termination_reason="error",
                    total_decompose_calls=decompose_calls,
                    total_retrieve_calls=retrieve_calls,
                )
            decompose = time.perf_counter() - decompose_t0

            logger.info(
                "iterative hop %d/%d: done=%s sub_q=%s",
                hop_index,
                iterative.max_hops,
                result.done,
                "<set>" if result.next_sub_question else "<none>",
            )

            if result.done:
                hop_traces.append(
                    IterativeHopTrace(
                        hop_index=hop_index,
                        sub_question=None,
                        findings_from_last_hop=result.findings_from_last_hop,
                        decompose_reasoning=result.reasoning,
                        timings={"decompose": decompose},
                    )
                )
                return accumulated_chunks, IterativeOutcome(
                    hops=hop_traces,
                    termination_reason="done",
                    total_decompose_calls=decompose_calls,
                    total_retrieve_calls=retrieve_calls,
                )

            sub_question = result.next_sub_question
            if not sub_question:
                # Decomposer violated contract: done=false but no sub_question
                # (or empty string). Treat as terminate-with-error rather than
                # spinning the loop on missing input.
                hop_traces.append(
                    IterativeHopTrace(
                        hop_index=hop_index,
                        sub_question=None,
                        findings_from_last_hop=result.findings_from_last_hop,
                        decompose_reasoning=(
                            f"contract violation: done=false with empty sub_question. "
                            f"{result.reasoning}"
                        ),
                        timings={"decompose": decompose},
                    )
                )
                return accumulated_chunks, IterativeOutcome(
                    hops=hop_traces,
                    termination_reason="error",
                    total_decompose_calls=decompose_calls,
                    total_retrieve_calls=retrieve_calls,
                )

            retrieve_calls += 1
            retrieve_t0 = time.perf_counter()
            # Per-hop retrieval. `min_score` threads through unchanged so
            # iterative respects the consumer's filter at the service
            # boundary just like single-pass retrieval. We pass
            # `trace=True` unconditionally — we need the inner trace data
            # to populate the hop's IterativeHopTrace; whether the *outer*
            # trace is surfaced is the engine arm's decision.
            hop_chunks, hop_trace_data = await self._retrieval_service.retrieve(
                query=sub_question,
                knowledge_id=knowledge_id,
                trace=True,
            )
            if min_score is not None:
                hop_chunks = [c for c in hop_chunks if c.score >= min_score]
            retrieve = time.perf_counter() - retrieve_t0

            accumulated_chunks = _merge_chunks_dedup(accumulated_chunks, hop_chunks)
            # The decomposer self-summarises; we replace, not append. This
            # bounds findings growth regardless of `max_hops` and matches
            # the prompt contract in R6.1's BAML function.
            accumulated_findings = result.findings_from_last_hop

            # R6.2: `expansion_applied` is structurally always False here.
            # Per-hop retrieval calls `RetrievalService.retrieve` directly,
            # bypassing `RagEngine._query_via_retrieval` where R5.3's
            # expansion loop sets the `expansion_attempts` key. Field
            # reserved for future per-hop expansion: when that work lands
            # it will populate the key and this read becomes truthful
            # without further changes here.
            expansion_applied = False
            adaptive_snapshot: dict[str, object] | None = None
            if hop_trace_data is not None and hop_trace_data.adaptive is not None:
                adaptive_snapshot = dict(hop_trace_data.adaptive)
                expansion_applied = bool(
                    adaptive_snapshot.get("expansion_attempts", 0)
                )

            hop_timings: dict[str, float] = {
                "decompose": decompose,
                "retrieve": retrieve,
            }
            if hop_trace_data is not None:
                hop_timings.update(hop_trace_data.timings)

            hop_traces.append(
                IterativeHopTrace(
                    hop_index=hop_index,
                    sub_question=sub_question,
                    findings_from_last_hop=result.findings_from_last_hop,
                    decompose_reasoning=result.reasoning,
                    per_method_results=(
                        dict(hop_trace_data.per_method_results) if hop_trace_data else {}
                    ),
                    fused_results=(
                        list(hop_trace_data.fused_results) if hop_trace_data else []
                    ),
                    reranked_results=(
                        list(hop_trace_data.reranked_results)
                        if hop_trace_data and hop_trace_data.reranked_results is not None
                        else None
                    ),
                    refined_results=(
                        list(hop_trace_data.refined_results)
                        if hop_trace_data and hop_trace_data.refined_results is not None
                        else None
                    ),
                    adaptive=adaptive_snapshot,
                    timings=hop_timings,
                    expansion_applied=expansion_applied,
                )
            )

        # Loop fell through: max_hops reached without `done=true`.
        return accumulated_chunks, IterativeOutcome(
            hops=hop_traces,
            termination_reason="max_hops",
            total_decompose_calls=decompose_calls,
            total_retrieve_calls=retrieve_calls,
        )