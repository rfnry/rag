from typing import Any, Protocol

from rfnry_rag.retrieval.common.models import SparseVector, VectorPoint, VectorResult


class BaseVectorStore(Protocol):
    async def initialize(self, vector_size: int) -> None: ...

    async def upsert(self, points: list[VectorPoint]) -> None: ...

    async def search(
        self,
        vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorResult]: ...

    async def hybrid_search(
        self,
        vector: list[float],
        sparse_vector: SparseVector,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorResult]: ...

    async def retrieve(self, point_ids: list[str]) -> list[VectorResult]: ...

    async def scroll(
        self,
        filters: dict[str, Any] | None = None,
        limit: int = 100,
        offset: str | None = None,
    ) -> tuple[list[VectorResult], str | None]: ...

    async def delete(self, filters: dict[str, Any]) -> int: ...

    async def count(self, filters: dict[str, Any] | None = None) -> int: ...

    async def shutdown(self) -> None: ...
