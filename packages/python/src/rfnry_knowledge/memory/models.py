from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class InteractionTurn:
    role: str
    content: str


@dataclass(frozen=True)
class Interaction:
    turns: tuple[InteractionTurn, ...]
    occurred_at: datetime | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExtractedMemory:
    text: str
    attributed_to: str | None
    linked_memory_row_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class MemoryRow:
    memory_row_id: str
    memory_id: str
    text: str
    text_hash: str
    attributed_to: str | None
    linked_memory_row_ids: tuple[str, ...]
    created_at: datetime
    updated_at: datetime
    interaction_metadata: Mapping[str, Any]


@dataclass(frozen=True)
class MemorySearchResult:
    row: MemoryRow
    score: float
    pillar_scores: Mapping[str, float]
