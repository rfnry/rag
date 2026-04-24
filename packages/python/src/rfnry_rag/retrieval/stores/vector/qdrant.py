from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    Fusion,
    FusionQuery,
    MatchAny,
    MatchValue,
    PointStruct,
    Prefetch,
    SparseVectorParams,
    VectorParams,
)
from qdrant_client.models import (
    SparseVector as QdrantSparseVector,
)

from rfnry_rag.retrieval.common.errors import ConfigurationError
from rfnry_rag.retrieval.common.logging import get_logger
from rfnry_rag.retrieval.common.models import SparseVector, VectorPoint, VectorResult

logger = get_logger(__name__)

_DENSE_VECTOR = "dense"
_SPARSE_VECTOR = "sparse"


@dataclass
class _CollectionState:
    initialized: bool = False
    named_vectors: bool = False


class QdrantVectorStore:
    """Qdrant vector store with multi-collection support.

    Accepts either a single ``collection`` name (backward compatible) or a
    ``collections`` list.  Every configured collection is verified / created
    during ``initialize()``.  All operation methods accept an optional
    ``collection`` parameter; when omitted they target the first collection
    in the list.

    Use ``scoped(name)`` to obtain a lightweight ``BaseVectorStore``-compatible
    wrapper that pins every operation to a single collection — ideal for
    passing to modules that don't know about multi-collection.
    """

    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection: str | None = None,
        collections: list[str] | None = None,
        timeout: int = 10,
        api_key: str | None = None,
        scroll_timeout: int = 30,
        write_timeout: int = 30,
        max_scroll_limit: int = 10_000,
        hybrid_prefetch_multiplier: int = 4,
    ) -> None:
        if hybrid_prefetch_multiplier < 1:
            raise ConfigurationError("hybrid_prefetch_multiplier must be >= 1")
        if timeout <= 0:
            raise ConfigurationError(f"timeout must be > 0, got {timeout}")
        if scroll_timeout <= 0:
            raise ConfigurationError(f"scroll_timeout must be > 0, got {scroll_timeout}")
        if write_timeout <= 0:
            raise ConfigurationError(f"write_timeout must be > 0, got {write_timeout}")
        self._url = url
        self._timeout = timeout
        self._scroll_timeout = scroll_timeout
        self._write_timeout = write_timeout
        self._max_scroll_limit = max_scroll_limit
        self._hybrid_prefetch_multiplier = hybrid_prefetch_multiplier
        self._api_key = api_key
        self._client_instance: AsyncQdrantClient | None = AsyncQdrantClient(
            url=self._url, timeout=timeout, api_key=api_key
        )

        if collections:
            self._collections = list(collections)
        elif collection:
            self._collections = [collection]
        else:
            self._collections = ["knowledge"]

        self._state: dict[str, _CollectionState] = {}

        if url.startswith("http://") and not api_key:
            logger.warning(
                "qdrant: plaintext HTTP with no API key at %s — do not use in production",
                url,
            )

    @property
    def _client(self) -> AsyncQdrantClient:
        assert self._client_instance is not None, "QdrantVectorStore has been shut down"
        return self._client_instance

    @property
    def collections(self) -> list[str]:
        """Configured collection names."""
        return self._collections

    async def initialize(self, vector_size: int) -> None:
        """Verify / create all configured collections."""
        response = await self._client.get_collections()
        existing = {c.name for c in response.collections}

        # SERIAL: self._state is mutated inside this loop; parallel iteration
        # would introduce a check-then-act race on the same dict. The loop body
        # is fast (one API call per collection) and initialize() is called once
        # at startup — the sequential cost is negligible.
        for name in self._collections:
            if name in self._state:
                continue
            if name in existing:
                info = await self._client.get_collection(name)
                named = isinstance(info.config.params.vectors, dict)
                self._state[name] = _CollectionState(initialized=True, named_vectors=named)
                logger.info("collection '%s' exists (named_vectors=%s)", name, named)
            else:
                await self._client.create_collection(
                    collection_name=name,
                    vectors_config={
                        _DENSE_VECTOR: VectorParams(size=vector_size, distance=Distance.COSINE),
                    },
                    sparse_vectors_config={
                        _SPARSE_VECTOR: SparseVectorParams(),
                    },
                )
                self._state[name] = _CollectionState(initialized=True, named_vectors=True)
                logger.info("created collection '%s' (dim=%d, named vectors)", name, vector_size)

        logger.info(
            "qdrant vector store initialized: url=%s timeout=%ds scroll_timeout=%ds write_timeout=%ds "
            "hybrid_prefetch_multiplier=%d collections=%s",
            self._url,
            self._timeout,
            self._scroll_timeout,
            self._write_timeout,
            self._hybrid_prefetch_multiplier,
            self._collections,
        )

    async def _ensure_and_resolve(self, collection: str | None = None) -> tuple[str, bool] | None:
        """Async version: ensure collection exists and return (name, named_vectors).

        Returns ``None`` when the collection does not exist.
        """
        name = collection or self._collections[0]
        state = self._state.get(name)
        if state:
            return name, state.named_vectors

        try:
            info = await self._client.get_collection(name)
        except Exception:
            return None
        named = isinstance(info.config.params.vectors, dict)
        self._state[name] = _CollectionState(initialized=True, named_vectors=named)
        return name, named

    def scoped(self, collection: str) -> _ScopedVectorStore:
        """Return a ``BaseVectorStore``-compatible wrapper pinned to *collection*."""
        if collection not in self._collections:
            raise ConfigurationError(f"Unknown collection: {collection!r}. Available: {self._collections}")
        return _ScopedVectorStore(parent=self, collection=collection)

    async def upsert(self, points: list[VectorPoint], collection: str | None = None) -> None:
        if not points:
            return

        resolved = await self._ensure_and_resolve(collection)
        if not resolved:
            raise ConfigurationError(f"Collection does not exist: {collection or self._collections[0]}")
        name, named = resolved

        if named:
            qdrant_points = []
            for p in points:
                vectors: dict[str, Any] = {_DENSE_VECTOR: p.vector}
                if p.sparse_vector:
                    vectors[_SPARSE_VECTOR] = QdrantSparseVector(
                        indices=p.sparse_vector.indices,
                        values=p.sparse_vector.values,
                    )
                qdrant_points.append(PointStruct(id=p.point_id, vector=vectors, payload=p.payload))
        else:
            qdrant_points = [PointStruct(id=p.point_id, vector=p.vector, payload=p.payload) for p in points]

        await self._client.upsert(
            collection_name=name,
            points=qdrant_points,
            timeout=self._write_timeout,
        )
        logger.info("upserted %d points to '%s'", len(points), name)

    async def search(
        self,
        vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        collection: str | None = None,
    ) -> list[VectorResult]:
        resolved = await self._ensure_and_resolve(collection)
        if not resolved:
            return []
        name, named = resolved
        query_filter = self._build_filter(filters)

        if named:
            response = await self._client.query_points(
                collection_name=name,
                query=vector,
                using=_DENSE_VECTOR,
                query_filter=query_filter,
                limit=top_k,
            )
        else:
            response = await self._client.query_points(
                collection_name=name,
                query=vector,
                query_filter=query_filter,
                limit=top_k,
            )

        return [VectorResult(point_id=str(r.id), score=r.score, payload=r.payload or {}) for r in response.points]

    async def hybrid_search(
        self,
        vector: list[float],
        sparse_vector: SparseVector,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        collection: str | None = None,
    ) -> list[VectorResult]:
        resolved = await self._ensure_and_resolve(collection)
        if not resolved:
            return []
        name, named = resolved

        if not named:
            return await self.search(vector=vector, top_k=top_k, filters=filters, collection=collection)

        query_filter = self._build_filter(filters)
        fetch_k = top_k * self._hybrid_prefetch_multiplier

        response = await self._client.query_points(
            collection_name=name,
            prefetch=[
                Prefetch(query=vector, using=_DENSE_VECTOR, limit=fetch_k, filter=query_filter),
                Prefetch(
                    query=QdrantSparseVector(indices=sparse_vector.indices, values=sparse_vector.values),
                    using=_SPARSE_VECTOR,
                    limit=fetch_k,
                    filter=query_filter,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
        )

        return [VectorResult(point_id=str(r.id), score=r.score, payload=r.payload or {}) for r in response.points]

    async def retrieve(self, point_ids: list[str], collection: str | None = None) -> list[VectorResult]:
        if not point_ids:
            return []
        resolved = await self._ensure_and_resolve(collection)
        if not resolved:
            return []
        name, _ = resolved

        points = await self._client.retrieve(
            collection_name=name,
            ids=point_ids,
            with_payload=True,
            with_vectors=False,
        )

        return [VectorResult(point_id=str(p.id), score=0.0, payload=p.payload or {}) for p in points]

    async def scroll(
        self,
        filters: dict[str, Any] | None = None,
        limit: int = 100,
        offset: str | None = None,
        collection: str | None = None,
    ) -> tuple[list[VectorResult], str | None]:
        resolved = await self._ensure_and_resolve(collection)
        if not resolved:
            return [], None
        name, _ = resolved
        query_filter = self._build_filter(filters)
        # Clamp caller-supplied limit to defend against unbounded scrolls.
        clamped_limit = min(max(1, limit), self._max_scroll_limit)

        points, next_offset = await self._client.scroll(
            collection_name=name,
            scroll_filter=query_filter,
            limit=clamped_limit,
            offset=offset,
            timeout=self._scroll_timeout,
        )

        results = [VectorResult(point_id=str(p.id), score=0.0, payload=p.payload or {}) for p in points]
        next_page = str(next_offset) if next_offset is not None else None
        return results, next_page

    async def delete(self, filters: dict[str, Any], collection: str | None = None) -> int:
        if not filters:
            return 0

        query_filter = self._build_filter(filters)
        if query_filter is None:
            return 0

        resolved = await self._ensure_and_resolve(collection)
        if not resolved:
            return 0
        name, _ = resolved
        count_before = await self.count(filters, collection=collection)

        await self._client.delete(
            collection_name=name,
            points_selector=FilterSelector(filter=query_filter),
            timeout=self._write_timeout,
        )

        logger.info("deleted %d points from '%s'", count_before, name)
        return count_before

    async def count(self, filters: dict[str, Any] | None = None, collection: str | None = None) -> int:
        resolved = await self._ensure_and_resolve(collection)
        if not resolved:
            return 0
        name, _ = resolved
        query_filter = self._build_filter(filters)

        result = await self._client.count(
            collection_name=name,
            count_filter=query_filter,
        )
        return result.count

    async def shutdown(self) -> None:
        if self._client_instance is not None:
            await self._client_instance.close()
            self._client_instance = None

    @staticmethod
    def _build_filter(filters: dict[str, Any] | None) -> Filter | None:
        if not filters:
            return None

        conditions: list[FieldCondition] = []
        for key, value in filters.items():
            if isinstance(value, list):
                conditions.append(FieldCondition(key=key, match=MatchAny(any=value)))
            else:
                conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))

        return Filter(must=conditions) if conditions else None  # type: ignore[arg-type]


class _ScopedVectorStore:
    """Lightweight wrapper that pins all operations to one collection.

    Satisfies the ``BaseVectorStore`` protocol so it can be handed to any
    module (RetrievalService, IngestionService, KeywordSearch) unchanged.
    """

    def __init__(self, parent: QdrantVectorStore, collection: str) -> None:
        self._parent = parent
        self._collection = collection

    async def initialize(self, vector_size: int) -> None:
        await self._parent.initialize(vector_size)

    async def upsert(self, points: list[VectorPoint]) -> None:
        await self._parent.upsert(points, collection=self._collection)

    async def search(
        self,
        vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorResult]:
        return await self._parent.search(vector, top_k, filters, collection=self._collection)

    async def hybrid_search(
        self,
        vector: list[float],
        sparse_vector: SparseVector,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorResult]:
        return await self._parent.hybrid_search(vector, sparse_vector, top_k, filters, collection=self._collection)

    async def retrieve(self, point_ids: list[str]) -> list[VectorResult]:
        return await self._parent.retrieve(point_ids, collection=self._collection)

    async def scroll(
        self,
        filters: dict[str, Any] | None = None,
        limit: int = 100,
        offset: str | None = None,
    ) -> tuple[list[VectorResult], str | None]:
        return await self._parent.scroll(filters, limit, offset, collection=self._collection)

    async def delete(self, filters: dict[str, Any]) -> int:
        return await self._parent.delete(filters, collection=self._collection)

    async def count(self, filters: dict[str, Any] | None = None) -> int:
        return await self._parent.count(filters, collection=self._collection)

    async def shutdown(self) -> None:
        pass
