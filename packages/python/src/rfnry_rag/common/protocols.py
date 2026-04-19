from __future__ import annotations

from typing import Any, Protocol


class BaseEmbeddings(Protocol):
    """Common embeddings contract used by both retrieval and reasoning SDKs."""

    @property
    def model(self) -> str: ...

    async def embed(self, texts: list[str]) -> list[list[float]]: ...

    async def embedding_dimension(self) -> int: ...


class BaseSemanticIndex(Protocol):
    """Read-only semantic lookup used by reasoning services that need vector search
    without the full retrieval VectorStore surface (upsert/delete/hybrid_search/etc)."""

    async def scroll(
        self,
        filters: dict[str, Any] | None = None,
        limit: int = 100,
        offset: str | None = None,
    ) -> tuple[list[Any], str | None]: ...

    async def search(
        self,
        vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[Any]: ...
