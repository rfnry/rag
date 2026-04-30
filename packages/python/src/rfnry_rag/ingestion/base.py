from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from rfnry_rag.ingestion.models import ChunkedContent, ParsedPage


class BaseIngestionMethod(Protocol):
    """Protocol for pluggable ingestion methods.

    Methods declare ``required = True`` when their failure must abort
    ingestion (no metadata commit, IngestionError raised). ``required = False``
    means the failure is logged and ingestion continues. If a method omits
    the attribute, the dispatcher treats it as required to preserve data
    integrity by default.
    """

    required: bool

    @property
    def name(self) -> str: ...

    async def ingest(
        self,
        source_id: str,
        knowledge_id: str | None,
        source_type: str | None,
        source_weight: float,
        title: str,
        full_text: str,
        chunks: list[ChunkedContent],
        tags: list[str],
        metadata: dict[str, Any],
        hash_value: str | None = None,
        pages: list[ParsedPage] | None = None,
    ) -> None: ...

    async def delete(self, source_id: str) -> None: ...


class PhasedIngestionMethod(Protocol):
    """Multi-phase ingestion methods (drawing, analyzed) — engine routes by isinstance."""

    required: bool

    @property
    def name(self) -> str: ...

    def accepts(self, file_path: Path, source_type: str | None) -> bool: ...

    async def delete(self, source_id: str) -> None: ...
