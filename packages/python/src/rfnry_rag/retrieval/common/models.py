from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal


@dataclass
class Source:
    source_id: str
    status: str = "completed"
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    chunk_count: int = 0
    embedding_model: str = ""
    file_hash: str | None = None
    stale: bool = False
    created_at: datetime | None = None
    knowledge_id: str | None = None
    source_type: str | None = None
    source_weight: float = 1.0

    @property
    def estimated_tokens(self) -> int | None:
        value = self.metadata.get("estimated_tokens")
        if value is None:
            return None
        return int(value)


@dataclass
class Chunk:
    chunk_id: str
    source_id: str
    content: str
    page_number: int | None = None
    section: str | None = None
    chunk_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None


@dataclass
class SparseVector:
    indices: list[int]
    values: list[float]


@dataclass
class VectorPoint:
    point_id: str
    vector: list[float]
    payload: dict[str, Any] = field(default_factory=dict)
    sparse_vector: SparseVector | None = None


@dataclass
class VectorResult:
    point_id: str
    score: float
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    chunk_id: str
    source_id: str
    content: str
    score: float
    page_number: int | None = None
    section: str | None = None
    source_type: str | None = None
    source_weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
    source_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalTrace:
    """Full per-query pipeline state for observability.

    `None` vs `[]` distinction is load-bearing: `None` means "stage did not
    run" (e.g. reranker disabled), `[]` means "stage ran and produced no
    results". Conflating them would erase signal.

    `per_method_results` is keyed by `BaseRetrievalMethod.name`. Across
    multiple query variants (multi-query rewriter), each method's per-variant
    results are concatenated — not deduplicated; fusion handles dedupe at
    the `fused_results` stage.

    `routing_decision` enumerates `"indexed" | "full_context" | "auto_indexed" | "auto_full_context"`.
    """

    query: str
    rewritten_queries: list[str] = field(default_factory=list)
    per_method_results: dict[str, list[RetrievedChunk]] = field(default_factory=dict)
    fused_results: list[RetrievedChunk] = field(default_factory=list)
    reranked_results: list[RetrievedChunk] | None = None
    refined_results: list[RetrievedChunk] | None = None
    final_results: list[RetrievedChunk] = field(default_factory=list)
    grounding_decision: str | None = None
    confidence: float | None = None
    routing_decision: str | None = None
    timings: dict[str, float] = field(default_factory=dict)
    knowledge_id: str | None = None


@dataclass
class ContentMatch:
    source_id: str
    title: str
    excerpt: str
    score: float
    match_type: Literal["fulltext", "exact"]
    source_type: str | None = None

    def __repr__(self) -> str:
        return f"ContentMatch(source_id={self.source_id!r}, score={self.score:.4f}, match_type={self.match_type!r})"


@dataclass
class SourceStats:
    source_id: str
    total_chunks: int = 0
    total_pages: int = 0
    avg_chunk_size: int = 0
    processing_time: float = 0.0
    total_hits: int = 0
    grounded_hits: int = 0
    ungrounded_hits: int = 0
