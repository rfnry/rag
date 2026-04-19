# src/rfnry-rag/retrieval/modules/retrieval/base.py
from __future__ import annotations

from typing import Any, Protocol

from rfnry_rag.retrieval.common.models import RetrievedChunk


class BaseRetrievalMethod(Protocol):
    """Protocol for pluggable retrieval methods."""

    @property
    def name(self) -> str: ...

    @property
    def weight(self) -> float: ...

    @property
    def top_k(self) -> int | None: ...

    async def search(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
        knowledge_id: str | None = None,
    ) -> list[RetrievedChunk]: ...
