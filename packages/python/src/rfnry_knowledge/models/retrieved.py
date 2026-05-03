from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


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
