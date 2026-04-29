"""IterativeHopTrace + IterativeOutcome — trace surface for multi-hop retrieval.

R6.1 introduced ``IterativeHopTrace`` (definition only); R6.2 adds
``IterativeOutcome`` (the service's return alongside accumulated chunks)
and ``IterativeRetrievalService`` populates both. R6.3 will extend the
``termination_reason`` union with ``"low_confidence_escalated"`` once
post-loop DIRECT escalation lands. Public surfaces live at
``retrieval.IterativeHopTrace`` and ``retrieval.IterativeOutcome``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rfnry_rag.retrieval.common.models import RetrievedChunk


@dataclass
class IterativeHopTrace:
    """One hop of the iterative retrieval loop.

    Final-decomposer-call hops (those that returned ``done=true``) carry empty
    ``per_method_results`` / ``fused_results`` lists; ``decompose_reasoning``
    explains why the loop stopped. ``sub_question`` is ``None`` on a stop hop
    and a string otherwise.

    ``adaptive`` mirrors ``RetrievalTrace.adaptive`` for the per-hop retrieve
    call. R5.2's classifier verdict + R5.3's expansion keys land here so a
    consumer debugging "what happened in hop 2?" can inspect the full
    adaptive context for that hop without reaching into the inner trace —
    keeping per-hop debug parity with single-pass retrieval.
    """

    hop_index: int
    sub_question: str | None
    findings_from_last_hop: str
    decompose_reasoning: str
    per_method_results: dict[str, list[RetrievedChunk]] = field(default_factory=dict)
    fused_results: list[RetrievedChunk] = field(default_factory=list)
    reranked_results: list[RetrievedChunk] | None = None
    refined_results: list[RetrievedChunk] | None = None
    adaptive: dict[str, object] | None = None
    timings: dict[str, float] = field(default_factory=dict)
    expansion_applied: bool = False


@dataclass
class IterativeOutcome:
    """Service-layer return value alongside the deduplicated chunks.

    ``termination_reason`` enumerates the loop's exit condition:

    - ``"done"`` — decomposer returned ``done=true`` (the gate or the loop
      decided no further hop was needed). Also returned with zero hops when
      the type-mode gate rejected the query at entry.
    - ``"max_hops"`` — exhausted ``max_hops`` without a ``done=true`` verdict.
    - ``"error"`` — decomposer violated the contract (returned ``done=false``
      with ``next_sub_question=None``); we terminate to avoid an infinite loop.

    R6.3 will add ``"low_confidence_escalated"`` once post-loop DIRECT
    escalation lands.
    """

    hops: list[IterativeHopTrace]
    termination_reason: str
    total_decompose_calls: int
    total_retrieve_calls: int
