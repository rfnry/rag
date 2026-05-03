from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


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
