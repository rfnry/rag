from __future__ import annotations

import time
from typing import Any

from rfnry_rag.ingestion.models import ChunkedContent, ParsedPage
from rfnry_rag.logging import get_logger
from rfnry_rag.observability.context import current_obs
from rfnry_rag.stores.document.base import BaseDocumentStore
from rfnry_rag.telemetry.context import current_ingest_row

logger = get_logger("ingestion.methods.document")


class DocumentIngestion:
    """Store full document text in the document store."""

    required: bool = True

    def __init__(self, store: BaseDocumentStore) -> None:
        self._store = store

    def clone_for_store(self, store: BaseDocumentStore) -> DocumentIngestion:
        return DocumentIngestion(store=store)

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
        notes: list[str] | None = None,
    ) -> None:
        start = time.perf_counter()
        obs = current_obs()
        row = current_ingest_row()
        try:
            await self._store.store_content(
                source_id=source_id,
                knowledge_id=knowledge_id,
                source_type=source_type,
                title=title,
                content=full_text,
            )
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            logger.info("stored in %dms", elapsed_ms)
            if row is not None:
                row.persist_ms += elapsed_ms
            if obs is not None:
                await obs.emit(
                    "info",
                    "ingestion.method.success",
                    f"{self.name} ingest ok",
                    method_name=self.name,
                    duration_ms=elapsed_ms,
                    source_id=source_id,
                )
        except Exception as exc:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            logger.warning("failed in %dms — %s", elapsed_ms, exc)
            if obs is not None:
                await obs.emit(
                    "error",
                    "ingestion.method.error",
                    f"{self.name} ingest failed",
                    method_name=self.name,
                    duration_ms=elapsed_ms,
                    source_id=source_id,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            raise

    async def delete(self, source_id: str) -> None:
        await self._store.delete_content(source_id)
