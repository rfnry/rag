"""IterativeHopTrace — per-hop trace dataclass for multi-hop retrieval (R6.1).

Definition only at R6.1; ``IterativeRetrievalService`` populates instances
in R6.2 once the hop loop lands. Public surface lives at the package root
(``retrieval.IterativeHopTrace``).
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
    """

    hop_index: int
    sub_question: str | None
    findings_from_last_hop: str
    decompose_reasoning: str
    per_method_results: dict[str, list[RetrievedChunk]] = field(default_factory=dict)
    fused_results: list[RetrievedChunk] = field(default_factory=list)
    reranked_results: list[RetrievedChunk] | None = None
    refined_results: list[RetrievedChunk] | None = None
    timings: dict[str, float] = field(default_factory=dict)
    expansion_applied: bool = False
