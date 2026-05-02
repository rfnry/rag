from __future__ import annotations

import time
from typing import Any

from rfnry_rag.logging import get_logger
from rfnry_rag.models import ContentMatch, RetrievedChunk
from rfnry_rag.observability.context import current_obs
from rfnry_rag.stores.document.base import BaseDocumentStore
from rfnry_rag.telemetry.context import current_query_row

logger = get_logger("retrieval.methods.document")


class DocumentRetrieval:
    """Full-text / substring search on documents via the document store."""

    def __init__(
        self,
        store: BaseDocumentStore,
        weight: float = 1.0,
        top_k: int | None = None,
    ) -> None:
        self._store = store
        self._weight = weight
        self._top_k = top_k

    def clone_for_store(self, store: BaseDocumentStore) -> DocumentRetrieval:
        return DocumentRetrieval(store=store, weight=self._weight, top_k=self._top_k)

    @property
    def name(self) -> str:
        return "document"

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
            matches = await self._store.search_content(query=query, knowledge_id=knowledge_id, top_k=top_k)
            results = self._convert(matches)
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            logger.info("%d results in %dms", len(results), elapsed_ms)
            if row is not None:
                row.method_durations_ms[self.name] = elapsed_ms
                if self.name not in row.methods_used:
                    row.methods_used.append(self.name)
                row.chunks_retrieved += len(results)
            if obs is not None:
                await obs.emit(
                    "retrieval.method.success",
                    f"{self.name} retrieval ok",
                    context={
                        "method_name": self.name,
                        "chunks": len(results),
                        "duration_ms": elapsed_ms,
                    },
                )
            return results
        except Exception as exc:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            logger.warning("failed in %dms — %s", elapsed_ms, exc)
            if row is not None:
                row.method_errors += 1
                row.method_durations_ms[self.name] = elapsed_ms
            if obs is not None:
                await obs.emit(
                    "retrieval.method.error",
                    f"{self.name} retrieval failed",
                    level="error",
                    context={"method_name": self.name, "duration_ms": elapsed_ms},
                    error=exc,
                )
            return []

    @staticmethod
    def _convert(matches: list[ContentMatch]) -> list[RetrievedChunk]:
        chunks = []
        for match in matches:
            chunks.append(
                RetrievedChunk(
                    chunk_id=f"fulltext:{match.source_id}",
                    source_id=match.source_id,
                    content=match.excerpt,
                    score=match.score,
                    source_type=match.source_type,
                    source_metadata={
                        "title": match.title,
                        "match_type": match.match_type,
                    },
                )
            )
        return chunks
