"""`BaseRetrievalMethod` adapter for tree-based retrieval.

Wraps `TreeSearchService` so it can be composed into a `RetrievalService` as
a plug-compatible `BaseRetrievalMethod`. The adapter loads tree indexes from
a metadata store and runs tree search per source.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import TYPE_CHECKING, Any

from rfnry_rag.retrieval.common.logging import get_logger
from rfnry_rag.retrieval.common.models import RetrievedChunk, TreeIndex
from rfnry_rag.retrieval.modules.ingestion.tree.toc import PageContent

if TYPE_CHECKING:
    from rfnry_rag.retrieval.modules.retrieval.tree.service import TreeSearchService
    from rfnry_rag.retrieval.stores.metadata.base import BaseMetadataStore

logger = get_logger("retrieval.methods.tree")


class TreeRetrieval:
    """Adapts `TreeSearchService` + metadata_store to the `BaseRetrievalMethod` protocol."""

    def __init__(
        self,
        service: TreeSearchService,
        metadata_store: BaseMetadataStore,
        weight: float = 1.0,
        top_k: int | None = None,
    ) -> None:
        self._service = service
        self._metadata_store = metadata_store
        self._weight = weight
        self._top_k = top_k

    @property
    def name(self) -> str:
        return "tree"

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
            sources = await self._metadata_store.list_sources(knowledge_id=knowledge_id)

            async def search_one(source: Any) -> list[RetrievedChunk]:
                tree_json = await self._metadata_store.get_tree_index(source.source_id)
                if not tree_json:
                    return []
                tree_index = TreeIndex.from_dict(json.loads(tree_json))
                if not tree_index.pages:
                    return []
                pages = [PageContent(index=p.index, text=p.text, token_count=p.token_count) for p in tree_index.pages]
                results = await self._service.search(query=query, tree_index=tree_index, pages=pages)
                if not results:
                    return []
                return self._service.to_retrieved_chunks(results, tree_index)

            per_source = await asyncio.gather(
                *(search_one(s) for s in sources),
                return_exceptions=True,
            )

            all_chunks: list[RetrievedChunk] = []
            for source, outcome in zip(sources, per_source, strict=True):
                if isinstance(outcome, BaseException):
                    logger.warning(
                        "tree retrieval for %s failed: %s — skipping", source.source_id, outcome
                    )
                    continue
                all_chunks.extend(outcome)

            elapsed = (time.perf_counter() - start) * 1000
            logger.info("%d results in %.1fms", len(all_chunks), elapsed)
            return all_chunks
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000
            logger.warning("failed in %.1fms — %s", elapsed, exc)
            return []
