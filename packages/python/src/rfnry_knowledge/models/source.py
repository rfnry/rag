from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


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

    @property
    def ingestion_notes(self) -> list[str]:
        notes = self.metadata.get("ingestion_notes")
        return list(notes) if isinstance(notes, list) else []

    @property
    def fully_ingested(self) -> bool:
        return not self.ingestion_notes


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
class RetrievalHealth:
    total_hits: int
    grounded_hits: int
    ungrounded_hits: int
    grounding_rate: float | None


@dataclass
class HealthSummary:
    source_id: str
    fully_ingested: bool
    ingestion_notes: list[str]
    stale_embedding: bool
    embedding_model: str
    retrieval: RetrievalHealth | None
