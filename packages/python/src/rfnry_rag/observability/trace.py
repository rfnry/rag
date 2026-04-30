from __future__ import annotations

from dataclasses import dataclass, field

from rfnry_rag.models.retrieved import RetrievedChunk


@dataclass
class RetrievalTrace:
    """Full per-query pipeline state for observability.

    `None` vs `[]` distinction is load-bearing: `None` means "stage did not
    run" (e.g. reranker disabled), `[]` means "stage ran and produced no
    results". Conflating them would erase signal.

    `per_method_results` is keyed by `BaseRetrievalMethod.name`.

    `routing_decision` enumerates `"indexed" | "full_context" | "auto_indexed" | "auto_full_context"`.
    """

    query: str
    per_method_results: dict[str, list[RetrievedChunk]] = field(default_factory=dict)
    fused_results: list[RetrievedChunk] = field(default_factory=list)
    reranked_results: list[RetrievedChunk] | None = None
    final_results: list[RetrievedChunk] = field(default_factory=list)
    grounding_decision: str | None = None
    confidence: float | None = None
    routing_decision: str | None = None
    timings: dict[str, float] = field(default_factory=dict)
    knowledge_id: str | None = None
