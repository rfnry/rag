from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

_MAX_TREE_DEPTH = 100
_MAX_TREE_NODES = 10_000
# _MAX_PDF_PAGES is 5_000; 100k is 20× generous — still catches a pathological
# tampered DB row that would OOM before TreeNode recursion guards trigger.
_MAX_TREE_PAGES = 100_000


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

    # Stored in `metadata["estimated_tokens"]` rather than a dedicated column so
    # R1.1 ships without a schema migration; legacy rows return None and are
    # lazy-computed by `KnowledgeManager.get_corpus_tokens`. Promote to a column
    # only if a real consumer needs to query/sort by token count.
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

    def __repr__(self) -> str:
        return f"SparseVector({len(self.indices)} non-zero entries)"


@dataclass
class VectorPoint:
    point_id: str
    vector: list[float]
    payload: dict[str, Any]
    sparse_vector: SparseVector | None = None

    def __repr__(self) -> str:
        sparse = f", sparse={len(self.sparse_vector.indices)} entries" if self.sparse_vector else ""
        return f"VectorPoint(point_id={self.point_id!r}, vector=[{len(self.vector)} dims]{sparse}, payload=...)"


@dataclass
class VectorResult:
    point_id: str
    score: float
    payload: dict[str, Any]

    def __repr__(self) -> str:
        return f"VectorResult(point_id={self.point_id!r}, score={self.score:.4f}, payload=...)"


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
    """Full per-query pipeline state for observability of vector retrieval.

    Without this, R1's AUTO routing and R5's adaptive weights are unobservable
    — tuning either blind wastes effort. Constructible with just `query=...`;
    every other field has a safe default so a partial trace can be filled
    progressively as stages run.

    `None` vs `[]` distinction matters and is load-bearing for downstream
    failure classification (R8.2): `None` means "stage did not run" (e.g.
    reranker disabled), while `[]` means "stage ran and produced no results".
    Conflating them would erase the signal R8.2's SCOPE_MISS / DRIFT
    classifiers depend on.

    `per_method_results` is keyed by `BaseRetrievalMethod.name` for configured
    methods, plus a synthetic `"tree"` key when tree-search results were
    merged in at the service layer (tree search is not a registered
    `BaseRetrievalMethod`). Across multiple query variants
    (HyDE / multi-query / step-back), each method's per-variant results are
    concatenated — not deduplicated; fusion handles dedupe at the
    `fused_results` stage.

    `adaptive` (R5.2) is `None` when the adaptive pipeline did not run (the
    default — `AdaptiveRetrievalConfig.enabled=False`); otherwise carries
    `complexity`, `query_type`, `effective_top_k`, `applied_multipliers`,
    `classification_source`. Asserting via `trace.adaptive["applied_multipliers"]`
    is the supported way for consumers / tests to inspect per-method weights
    without reaching into service internals.
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
    adaptive: dict[str, Any] | None = None
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


@dataclass
class TreeNode:
    """A node in the document tree index."""

    node_id: str
    title: str
    start_index: int
    end_index: int
    summary: str | None = None
    children: list[TreeNode] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "title": self.title,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "summary": self.summary,
            "children": [c.to_dict() for c in self.children],
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        _depth: int = 0,
        _node_count: list[int] | None = None,
    ) -> TreeNode:
        if _depth > _MAX_TREE_DEPTH:
            raise ValueError(f"tree index depth exceeds {_MAX_TREE_DEPTH}")
        if _node_count is None:
            _node_count = [0]
        _node_count[0] += 1
        if _node_count[0] > _MAX_TREE_NODES:
            raise ValueError(f"tree index node count exceeds {_MAX_TREE_NODES}")
        return cls(
            node_id=data["node_id"],
            title=data["title"],
            start_index=data["start_index"],
            end_index=data["end_index"],
            summary=data.get("summary"),
            children=[cls.from_dict(c, _depth + 1, _node_count) for c in data.get("children", [])],
        )


@dataclass
class TreePage:
    """A page stored alongside the tree index for query-time retrieval."""

    index: int
    text: str
    token_count: int

    def to_dict(self) -> dict[str, Any]:
        return {"index": self.index, "text": self.text, "token_count": self.token_count}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TreePage:
        return cls(index=data["index"], text=data["text"], token_count=data["token_count"])


@dataclass
class TreeIndex:
    """Complete tree index for a document."""

    source_id: str
    doc_name: str
    doc_description: str | None
    structure: list[TreeNode]
    page_count: int
    created_at: datetime
    pages: list[TreePage] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "doc_name": self.doc_name,
            "doc_description": self.doc_description,
            "structure": [n.to_dict() for n in self.structure],
            "page_count": self.page_count,
            "created_at": self.created_at.isoformat(),
            "pages": [p.to_dict() for p in self.pages],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TreeIndex:
        pages_raw = data.get("pages", [])
        if len(pages_raw) > _MAX_TREE_PAGES:
            raise ValueError(
                f"tree index pages count {len(pages_raw)} exceeds {_MAX_TREE_PAGES}"
            )
        return cls(
            source_id=data["source_id"],
            doc_name=data["doc_name"],
            doc_description=data.get("doc_description"),
            structure=[TreeNode.from_dict(n) for n in data["structure"]],
            page_count=data["page_count"],
            created_at=datetime.fromisoformat(data["created_at"]),
            pages=[TreePage.from_dict(p) for p in pages_raw],
        )


@dataclass
class TreeSearchResult:
    """Result from tree-based search."""

    node_id: str
    title: str
    pages: str
    content: str
    reasoning: str
