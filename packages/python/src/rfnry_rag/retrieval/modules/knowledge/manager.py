from __future__ import annotations

import builtins
from collections.abc import Awaitable, Callable
from typing import Any

from rfnry_rag.retrieval.common.logging import get_logger
from rfnry_rag.retrieval.common.models import Chunk, Source, SourceStats
from rfnry_rag.retrieval.stores.document.base import BaseDocumentStore
from rfnry_rag.retrieval.stores.graph.base import BaseGraphStore
from rfnry_rag.retrieval.stores.metadata.base import BaseMetadataStore
from rfnry_rag.retrieval.stores.vector.base import BaseVectorStore

logger = get_logger("knowledge")


class KnowledgeManager:
    def __init__(
        self,
        vector_store: BaseVectorStore | None = None,
        metadata_store: BaseMetadataStore | None = None,
        on_source_removed: Callable[[str | None], Awaitable[None]] | None = None,
        document_store: BaseDocumentStore | None = None,
        graph_store: BaseGraphStore | None = None,
    ) -> None:
        self._vector_store = vector_store
        self._metadata_store = metadata_store
        self._on_source_removed = on_source_removed
        self._document_store = document_store
        self._graph_store = graph_store

    async def list(self, knowledge_id: str | None = None) -> builtins.list[Source]:
        """List all sources, optionally filtered by knowledge_id."""
        if self._metadata_store:
            return await self._metadata_store.list_sources(knowledge_id=knowledge_id)

        return await self._aggregate_sources_from_vectors(knowledge_id=knowledge_id)

    async def get(self, source_id: str) -> Source | None:
        """Get a single source by ID."""
        if not source_id or not source_id.strip():
            raise ValueError("source_id must not be empty")
        if self._metadata_store:
            return await self._metadata_store.get_source(source_id)

        sources = await self._aggregate_sources_from_vectors(source_id=source_id)
        return sources[0] if sources else None

    async def get_chunks(self, source_id: str) -> builtins.list[Chunk]:
        """Inspect all chunks belonging to a source."""
        if not source_id or not source_id.strip():
            raise ValueError("source_id must not be empty")
        if not self._vector_store:
            return []
        chunks: list[Chunk] = []
        offset = None

        while True:
            results, next_offset = await self._vector_store.scroll(
                filters={"source_id": source_id},
                limit=100,
                offset=offset,
            )
            for r in results:
                chunks.append(
                    Chunk(
                        chunk_id=r.point_id,
                        source_id=source_id,
                        content=r.payload.get("content", ""),
                        page_number=r.payload.get("page_number"),
                        section=r.payload.get("section"),
                        chunk_index=r.payload.get("chunk_index", 0),
                        metadata={
                            "tags": r.payload.get("tags", []),
                            "source_name": r.payload.get("source_name", ""),
                        },
                    )
                )
            if next_offset is None or not results:
                break
            offset = next_offset

        chunks.sort(key=lambda c: c.chunk_index)
        return chunks

    async def get_stats(self, source_id: str) -> SourceStats | None:
        """Get hit/grounding statistics for a source."""
        if not source_id or not source_id.strip():
            raise ValueError("source_id must not be empty")
        if self._metadata_store:
            return await self._metadata_store.get_source_stats(source_id)
        return None

    async def list_stale(
        self, knowledge_id: str | None = None
    ) -> builtins.list[Source]:
        """List sources whose stored embedding model differs from the current config.

        Stale sources produce embeddings that don't compare meaningfully against
        the current query embeddings. They should be re-ingested or removed.
        Requires a metadata store."""
        if not self._metadata_store:
            return []
        sources = await self._metadata_store.list_sources(knowledge_id=knowledge_id)
        return [s for s in sources if s.stale]

    async def purge_stale(self, knowledge_id: str | None = None) -> int:
        """Remove all stale sources. Returns the count of removed sources."""
        stale = await self.list_stale(knowledge_id=knowledge_id)
        for source in stale:
            await self.remove(source.source_id)
        return len(stale)

    async def remove(self, source_id: str) -> int:
        """Delete a source and all its chunks from both stores."""
        if not source_id or not source_id.strip():
            raise ValueError("source_id must not be empty")

        knowledge_id: str | None = None
        if self._metadata_store:
            source = await self._metadata_store.get_source(source_id)
            if source:
                knowledge_id = source.knowledge_id

        deleted = 0
        if self._vector_store:
            deleted = await self._vector_store.delete(filters={"source_id": source_id})

        if self._metadata_store:
            await self._metadata_store.delete_source(source_id)

        if self._document_store:
            await self._document_store.delete_content(source_id)

        if self._graph_store:
            await self._graph_store.delete_by_source(source_id)

        if self._on_source_removed:
            await self._on_source_removed(knowledge_id)

        logger.info("removed source %s: %d vectors deleted", source_id, deleted)
        return deleted

    async def _aggregate_sources_from_vectors(
        self,
        source_id: str | None = None,
        knowledge_id: str | None = None,
    ) -> builtins.list[Source]:
        """Derive source list by scanning vector payloads."""
        if not self._vector_store:
            return []
        sources_map: dict[str, dict[str, Any]] = {}
        offset = None
        filters: dict[str, Any] | None = None
        if source_id:
            filters = {"source_id": source_id}
        elif knowledge_id:
            filters = {"knowledge_id": knowledge_id}

        while True:
            results, next_offset = await self._vector_store.scroll(
                filters=filters,
                limit=500,
                offset=offset,
            )
            for r in results:
                sid = r.payload.get("source_id", "")
                if sid not in sources_map:
                    sources_map[sid] = {
                        "id": sid,
                        "name": r.payload.get("source_name", ""),
                        "file_url": r.payload.get("file_url", ""),
                        "tags": r.payload.get("tags", []),
                        "knowledge_id": r.payload.get("knowledge_id"),
                        "source_type": r.payload.get("source_type"),
                        "source_weight": r.payload.get("source_weight", 1.0),
                        "chunk_count": 0,
                    }
                sources_map[sid]["chunk_count"] += 1

            if next_offset is None or not results:
                break
            offset = next_offset

        return [
            Source(
                source_id=data["id"],
                metadata={"name": data["name"], "file_url": data["file_url"]},
                tags=data["tags"],
                chunk_count=data["chunk_count"],
                embedding_model="unknown",
                file_hash=None,
                created_at=None,
                stale=False,
                knowledge_id=data["knowledge_id"],
                source_type=data["source_type"],
                source_weight=data["source_weight"],
            )
            for data in sources_map.values()
        ]
