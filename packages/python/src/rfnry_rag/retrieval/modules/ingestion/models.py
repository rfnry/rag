from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class ParsedPage:
    page_number: int
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChunkedContent:
    content: str
    page_number: int | None = None
    section: str | None = None
    chunk_index: int = 0
    context: str = ""
    contextualized: str = ""
    parent_id: str | None = None
    chunk_type: Literal["child", "parent"] = "child"

    @property
    def embedding_text(self) -> str:
        """Text to use for embedding — contextualized if available, otherwise raw content."""
        return self.contextualized or self.content
