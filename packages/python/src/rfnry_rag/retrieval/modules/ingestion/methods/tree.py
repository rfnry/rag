from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from rfnry_rag.retrieval.common.logging import get_logger
from rfnry_rag.retrieval.modules.ingestion.models import ChunkedContent, ParsedPage

if TYPE_CHECKING:
    from rfnry_rag.retrieval.modules.ingestion.tree.service import TreeIndexingService

logger = get_logger("ingestion.methods.tree")


class TreeIngestion:
    """Build and persist tree index from parsed pages."""

    def __init__(self, tree_service: TreeIndexingService) -> None:
        self._service = tree_service

    @property
    def name(self) -> str:
        return "tree"

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
        if not pages:
            logger.info("skipped — no pages provided")
            return
        start = time.perf_counter()
        try:
            from rfnry_rag.retrieval.modules.ingestion.tree.toc import PageContent

            page_contents = [
                PageContent(index=p.page_number, text=p.content, token_count=len(p.content) // 4) for p in pages
            ]
            tree_idx = await self._service.build_tree_index(
                source_id=source_id,
                doc_name=title,
                pages=page_contents,
            )
            await self._service.save_tree_index(tree_idx)
            elapsed = (time.perf_counter() - start) * 1000
            logger.info("built tree index (%d pages) in %.1fms", len(pages), elapsed)
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000
            logger.warning("failed in %.1fms — %s", elapsed, exc)

    async def delete(self, source_id: str) -> None:
        pass
