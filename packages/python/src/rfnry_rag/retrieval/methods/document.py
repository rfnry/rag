from __future__ import annotations

import time
from typing import Any

from rfnry_rag.logging import get_logger
from rfnry_rag.models import ContentMatch, RetrievedChunk
from rfnry_rag.stores.document.base import BaseDocumentStore

logger = get_logger("retrieval.methods.document")


class DocumentRetrieval:
    """Full-text / substring search on documents via the document store."""

    def __init__(
        self,
        document_store: BaseDocumentStore,
        weight: float = 1.0,
        top_k: int | None = None,
    ) -> None:
        self._store = document_store
        self._weight = weight
        self._top_k = top_k

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
        try:
            matches = await self._store.search_content(query=query, knowledge_id=knowledge_id, top_k=top_k)
            results = self._convert(matches)
            elapsed = (time.perf_counter() - start) * 1000
            logger.info("%d results in %.1fms", len(results), elapsed)
            return results
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000
            logger.warning("failed in %.1fms — %s", elapsed, exc)
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
