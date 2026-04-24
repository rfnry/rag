import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import Any

from rank_bm25 import BM25Okapi

from rfnry_rag.retrieval.common.logging import get_logger
from rfnry_rag.retrieval.common.models import RetrievedChunk
from rfnry_rag.retrieval.stores.vector.base import BaseVectorStore

logger = get_logger("search/retrieval/bm25")

_GLOBAL_KEY = "__global__"


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


@dataclass
class _BM25Entry:
    index: BM25Okapi | None
    chunks: list[dict[str, Any]] = field(default_factory=list)
    last_used: float = 0.0


class KeywordSearch:
    def __init__(self, vector_store: BaseVectorStore, max_indexes: int = 16) -> None:
        self._store = vector_store
        self._max_indexes = max_indexes
        self._cache: dict[str, _BM25Entry] = {}
        self._lock = asyncio.Lock()

    async def invalidate(self, knowledge_id: str | None = None) -> None:
        key = knowledge_id if knowledge_id is not None else _GLOBAL_KEY
        async with self._lock:
            self._cache.pop(key, None)

    def _evict_lru(self) -> None:
        if len(self._cache) < self._max_indexes:
            return
        oldest_key = min(self._cache, key=lambda k: self._cache[k].last_used)
        del self._cache[oldest_key]
        logger.info("evicted bm25 index for key=%s (lru)", oldest_key)

    async def _build_index(self, knowledge_id: str | None) -> None:
        key = knowledge_id if knowledge_id is not None else _GLOBAL_KEY

        # Fast path: cache hit, no work.
        async with self._lock:
            if key in self._cache:
                return

        # Collect chunks OUTSIDE the lock. Multiple builds for different
        # keys can proceed concurrently; the vector store is the natural
        # serialization point.
        filters = {"knowledge_id": knowledge_id} if knowledge_id is not None else None
        all_chunks: list[dict[str, Any]] = []
        offset = None

        while True:
            results, next_offset = await self._store.scroll(filters=filters, limit=500, offset=offset)
            for r in results:
                all_chunks.append(
                    {
                        "point_id": r.point_id,
                        "content": r.payload.get("content", ""),
                        "page_number": r.payload.get("page_number"),
                        "section": r.payload.get("section"),
                        "source_id": r.payload.get("source_id", ""),
                        "source_type": r.payload.get("source_type"),
                        "source_weight": r.payload.get("source_weight", 1.0),
                        "source_name": r.payload.get("source_name", ""),
                        "file_url": r.payload.get("file_url", ""),
                        "tags": r.payload.get("tags", []),
                    }
                )
            if next_offset is None or not results:
                break
            offset = next_offset

        # Tokenize + build in thread pool — pure-Python CPU work.
        loop = asyncio.get_running_loop()
        if all_chunks:
            tokenized = await loop.run_in_executor(None, lambda: [_tokenize(c["content"]) for c in all_chunks])
            index: BM25Okapi | None = await loop.run_in_executor(None, lambda: BM25Okapi(tokenized))
        else:
            index = None

        # Re-check-then-write under the lock. If another build landed first,
        # discard ours.
        async with self._lock:
            if key in self._cache:
                return
            self._evict_lru()
            self._cache[key] = _BM25Entry(index=index, chunks=all_chunks, last_used=time.monotonic())
        logger.info("built bm25 index for knowledge_id=%s: %d chunks", knowledge_id, len(all_chunks))

    async def search(
        self,
        query: str,
        top_k: int = 10,
        knowledge_id: str | None = None,
    ) -> list[RetrievedChunk]:
        key = knowledge_id if knowledge_id is not None else _GLOBAL_KEY
        if key not in self._cache:
            await self._build_index(knowledge_id)

        entry = self._cache.get(key)
        if entry is None or entry.index is None or not entry.chunks:
            return []

        entry.last_used = time.monotonic()

        tokenized_query = _tokenize(query)
        scores = entry.index.get_scores(tokenized_query)

        scored = sorted(
            zip(scores, entry.chunks, strict=True),
            key=lambda x: x[0],
            reverse=True,
        )

        results = []
        for score, chunk in scored:
            if score <= 0:
                break
            results.append(
                RetrievedChunk(
                    chunk_id=chunk["point_id"],
                    content=chunk["content"],
                    score=float(score),
                    page_number=chunk["page_number"],
                    section=chunk["section"],
                    source_id=chunk["source_id"],
                    source_type=chunk["source_type"],
                    source_weight=chunk["source_weight"],
                    source_metadata={
                        "name": chunk["source_name"],
                        "file_url": chunk["file_url"],
                        "tags": chunk["tags"],
                    },
                )
            )
            if len(results) >= top_k:
                break

        logger.info("%d candidates from keyword search", len(results))
        return results
