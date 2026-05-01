from __future__ import annotations

import builtins
from collections.abc import Awaitable, Callable
from typing import Any

from rfnry_rag.ingestion.chunk.token_counter import count_tokens
from rfnry_rag.logging import get_logger
from rfnry_rag.models import Chunk, HealthSummary, RetrievalHealth, Source, SourceStats
from rfnry_rag.stores.document.base import BaseDocumentStore
from rfnry_rag.stores.graph.base import BaseGraphStore
from rfnry_rag.stores.metadata.base import BaseMetadataStore
from rfnry_rag.stores.vector.base import BaseVectorStore

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

    async def health(self, source_id: str) -> HealthSummary | None:
        """Fuse ingestion notes, retrieval stats, and embedding freshness for a source."""
        if not source_id or not source_id.strip():
            raise ValueError("source_id must not be empty")
        source = await self.get(source_id)
        if source is None:
            return None
        stats = await self.get_stats(source_id)
        retrieval: RetrievalHealth | None = None
        if stats is not None:
            grounding_rate = stats.grounded_hits / stats.total_hits if stats.total_hits else None
            retrieval = RetrievalHealth(
                total_hits=stats.total_hits,
                grounded_hits=stats.grounded_hits,
                ungrounded_hits=stats.ungrounded_hits,
                grounding_rate=grounding_rate,
            )
        return HealthSummary(
            source_id=source.source_id,
            fully_ingested=source.fully_ingested,
            ingestion_notes=source.ingestion_notes,
            stale_embedding=source.stale,
            embedding_model=source.embedding_model,
            retrieval=retrieval,
        )

    async def list_stale(self, knowledge_id: str | None = None) -> builtins.list[Source]:
        """List sources whose stored embedding model differs from the current config.

        Stale sources produce embeddings that don't compare meaningfully against
        the current query embeddings. They should be re-ingested or removed.
        Requires a metadata store."""
        if not self._metadata_store:
            return []
        sources = await self._metadata_store.list_sources(knowledge_id=knowledge_id)
        return [s for s in sources if s.stale]

    async def get_corpus_tokens(self, knowledge_id: str | None = None) -> int:
        """Sum estimated token counts across every source in scope.

        Reads `Source.estimated_tokens` (backed by `metadata["estimated_tokens"]`)
        when populated. Legacy sources ingested before token counting was
        added lack the count; for those we lazy-compute by reading source
        text from the document store and write the result back via
        `update_source(metadata=...)` so subsequent calls short-circuit.
        Vector-scroll fallback for legacy sources without
        a document store is intentionally NOT implemented here — the legacy
        path is rare and adding scroll-based reconstruction here would
        duplicate `RagEngine._load_full_corpus`.
        """
        if not self._metadata_store:
            return 0

        sources = await self._metadata_store.list_sources(knowledge_id=knowledge_id)
        # Deterministic order regardless of metadata-store sort. Helps with
        # concurrency, test fixture stability, and keeps the DIRECT-mode
        # prompt-caching prefix pinned to source_id rather than created_at
        # (mirrors the same sort applied in `RagEngine._load_full_corpus`).
        sources = sorted(sources, key=lambda s: s.source_id)
        total = 0
        # Sequential per-source: lazy-compute touches the document store + the
        # metadata store per legacy row, which is rare in practice. Batching is
        # straightforward to add later if a hot-path consumer needs it.
        for source in sources:
            cached = source.estimated_tokens
            if cached is not None:
                total += cached
                continue

            text = ""
            if self._document_store is not None:
                text = await self._document_store.get(source.source_id) or ""
            token_count = count_tokens(text) if text else 0
            # Only writeback when we actually computed something. Writing a
            # zero would poison the cache forever — `cached is not None` above
            # would short-circuit subsequent calls even after a document store
            # is later configured or the source is re-ingested. Empty-text
            # legacy sources stay un-cached so retries can succeed.
            if text and token_count > 0:
                source.metadata["estimated_tokens"] = token_count
                await self._metadata_store.update_source(
                    source.source_id,
                    metadata=source.metadata,
                )
            total += token_count
        return total

    async def purge_stale(self, knowledge_id: str | None = None) -> int:
        """Remove all stale sources. Returns the count of removed sources."""
        stale = await self.list_stale(knowledge_id=knowledge_id)
        # SERIAL: remove() touches the metadata store, vector store, document
        # store, and graph store for each source. Running removes concurrently
        # can produce conflicting deletes on shared indexes and raises in the
        # metadata store if two coroutines attempt FK-constrained deletes at once.
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
