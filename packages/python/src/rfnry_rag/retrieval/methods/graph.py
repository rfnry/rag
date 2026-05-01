from __future__ import annotations

import time
from typing import Any

from rfnry_rag.logging import get_logger
from rfnry_rag.models import RetrievedChunk
from rfnry_rag.observability.context import current_obs
from rfnry_rag.stores.graph.base import BaseGraphStore
from rfnry_rag.stores.graph.models import GraphPath, GraphResult
from rfnry_rag.telemetry.context import current_query_row

logger = get_logger("retrieval.methods.graph")


class GraphRetrieval:
    """Entity lookup + N-hop graph traversal via the graph store."""

    def __init__(
        self,
        store: BaseGraphStore,
        weight: float = 1.0,
        top_k: int | None = None,
    ) -> None:
        self._store = store
        self._weight = weight
        self._top_k = top_k

    def clone_for_store(self, store: BaseGraphStore) -> GraphRetrieval:
        return GraphRetrieval(store=store, weight=self._weight, top_k=self._top_k)

    @property
    def name(self) -> str:
        return "graph"

    @property
    def weight(self) -> float:
        return self._weight

    @property
    def top_k(self) -> int | None:
        return self._top_k

    async def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        knowledge_id: str | None = None,
    ) -> list[RetrievedChunk]:
        start = time.perf_counter()
        obs = current_obs()
        row = current_query_row()
        try:
            results = await self._store.query_graph(query=query, knowledge_id=knowledge_id, max_hops=2, top_k=top_k)
            chunks = self._convert(results)
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            logger.info("%d results in %dms", len(chunks), elapsed_ms)
            if row is not None:
                row.method_durations_ms[self.name] = elapsed_ms
                if self.name not in row.methods_used:
                    row.methods_used.append(self.name)
                row.chunks_retrieved += len(chunks)
            if obs is not None:
                await obs.emit(
                    "info",
                    "retrieval.method.success",
                    f"{self.name} retrieval ok",
                    method_name=self.name,
                    chunks=len(chunks),
                    duration_ms=elapsed_ms,
                )
            return chunks
        except Exception as exc:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            logger.warning("failed in %dms — %s", elapsed_ms, exc)
            if row is not None:
                row.method_errors += 1
                row.method_durations_ms[self.name] = elapsed_ms
            if obs is not None:
                await obs.emit(
                    "error",
                    "retrieval.method.error",
                    f"{self.name} retrieval failed",
                    method_name=self.name,
                    duration_ms=elapsed_ms,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            return []

    async def trace(
        self,
        entity_name: str,
        max_hops: int = 2,
        relation_types: list[str] | None = None,
        knowledge_id: str | None = None,
    ) -> list[GraphPath]:
        """Traverse the knowledge graph from entity_name, returning all matching paths.

        This is a thin wrapper around the graph store's query_graph() + N-hop
        traversal. Unlike search(), it returns GraphPath objects directly so
        callers can inspect the actual connectivity subgraph without re-parsing
        a RetrievedChunk.description string.

        When relation_types is provided, only paths whose every edge is in the
        list are returned (strict AND filter).

        max_hops bounds the traversal depth and is passed through to the store.
        Errors from the store are treated the same way search() treats them:
        logged at warning level and converted to an empty result list.
        """
        start = time.perf_counter()
        try:
            results = await self._store.query_graph(
                query=entity_name,
                knowledge_id=knowledge_id,
                max_hops=max_hops,
                top_k=self._top_k or 10,
            )
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000
            logger.warning("trace(%r) failed in %.1fms — %s", entity_name, elapsed, exc)
            return []

        paths: list[GraphPath] = []
        for r in results:
            paths.extend(r.paths)
        if relation_types is not None:
            allowed = set(relation_types)
            paths = [p for p in paths if all(rel in allowed for rel in p.relationships)]
        elapsed = (time.perf_counter() - start) * 1000
        logger.info("trace(%r) returned %d paths in %.1fms", entity_name, len(paths), elapsed)
        return paths

    @staticmethod
    def _convert(results: list[GraphResult]) -> list[RetrievedChunk]:
        chunks: list[RetrievedChunk] = []
        for result in results:
            lines = [f"{result.entity.name} ({result.entity.entity_type})"]
            if result.entity.value:
                lines.append(f"  Specifications: {result.entity.value}")

            for path in result.paths:
                parts: list[str] = []
                for i, entity_name in enumerate(path.entities):
                    if i > 0 and i - 1 < len(path.relationships):
                        parts.append(f"-[{path.relationships[i - 1]}]->")
                    parts.append(entity_name)
                lines.append(f"  Path: {' '.join(parts)}")

            for connected in result.connected_entities[:5]:
                lines.append(f"  Connected: {connected.name} ({connected.entity_type})")

            chunks.append(
                RetrievedChunk(
                    chunk_id=f"graph:{result.entity.name}:{result.entity.entity_type}",
                    source_id=result.entity.properties.get("source_id", ""),
                    content="\n".join(lines),
                    score=result.relevance_score,
                    source_metadata={
                        "retrieval_type": "graph",
                        "entity_name": result.entity.name,
                        "entity_type": result.entity.entity_type,
                        "connected_count": len(result.connected_entities),
                    },
                )
            )
        return chunks
