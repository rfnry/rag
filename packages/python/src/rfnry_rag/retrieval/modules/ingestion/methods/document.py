from __future__ import annotations

import time
from typing import Any

from rfnry_rag.retrieval.common.logging import get_logger
from rfnry_rag.retrieval.modules.ingestion.models import ChunkedContent, ParsedPage
from rfnry_rag.retrieval.stores.document.base import BaseDocumentStore

logger = get_logger("ingestion.methods.document")


class DocumentIngestion:
    """Store full document text in the document store."""

    def __init__(self, document_store: BaseDocumentStore) -> None:
        self._store = document_store

    @property
    def name(self) -> str:
        return "document"

    async def ingest(
        self,
        source_id: str,
        knowledge_id: str | None,
        source_type: str | None,
        source_weight: float,
        title: str,
        full_text: str,
        chunks: list[ChunkedContent],
        tags: list[str],
        metadata: dict[str, Any],
        hash_value: str | None = None,
        pages: list[ParsedPage] | None = None,
    ) -> None:
        start = time.perf_counter()
        try:
            await self._store.store_content(
                source_id=source_id,
                knowledge_id=knowledge_id,
                source_type=source_type,
                title=title,
                content=full_text,
            )
            elapsed = (time.perf_counter() - start) * 1000
            logger.info("stored in %.1fms", elapsed)
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000
            logger.warning("failed in %.1fms — %s", elapsed, exc)
            raise

    async def delete(self, source_id: str) -> None:
        await self._store.delete_content(source_id)
