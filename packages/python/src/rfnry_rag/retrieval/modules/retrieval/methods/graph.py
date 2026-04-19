from __future__ import annotations

import time
from typing import Any

from rfnry_rag.retrieval.common.logging import get_logger
from rfnry_rag.retrieval.common.models import RetrievedChunk
from rfnry_rag.retrieval.stores.graph.base import BaseGraphStore
from rfnry_rag.retrieval.stores.graph.models import GraphResult

logger = get_logger("retrieval.methods.graph")


class GraphRetrieval:
    """Entity lookup + N-hop graph traversal via the graph store."""

    def __init__(
        self,
        graph_store: BaseGraphStore,
        weight: float = 1.0,
        top_k: int | None = None,
    ) -> None:
        self._store = graph_store
        self._weight = weight
        self._top_k = top_k

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
        try:
            results = await self._store.query_graph(query=query, knowledge_id=knowledge_id, max_hops=2, top_k=top_k)
            chunks = self._convert(results)
            elapsed = (time.perf_counter() - start) * 1000
            logger.info("%d results in %.1fms", len(chunks), elapsed)
            return chunks
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000
            logger.warning("failed in %.1fms — %s", elapsed, exc)
            return []

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
