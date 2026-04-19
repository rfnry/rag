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
    def from_dict(cls, data: dict[str, Any]) -> TreeNode:
        return cls(
            node_id=data["node_id"],
            title=data["title"],
            start_index=data["start_index"],
            end_index=data["end_index"],
            summary=data.get("summary"),
            children=[cls.from_dict(c) for c in data.get("children", [])],
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
        return cls(
            source_id=data["source_id"],
            doc_name=data["doc_name"],
            doc_description=data.get("doc_description"),
            structure=[TreeNode.from_dict(n) for n in data["structure"]],
            page_count=data["page_count"],
            created_at=datetime.fromisoformat(data["created_at"]),
            pages=[TreePage.from_dict(p) for p in data.get("pages", [])],
        )


@dataclass
class TreeSearchResult:
    """Result from tree-based search."""

    node_id: str
    title: str
    pages: str
    content: str
    reasoning: str
