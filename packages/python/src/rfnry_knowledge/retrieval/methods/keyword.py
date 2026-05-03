from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from rank_bm25 import BM25Okapi

from rfnry_knowledge.common.logging import get_logger
from rfnry_knowledge.exceptions import ConfigurationError
from rfnry_knowledge.models import ContentMatch, RetrievedChunk
from rfnry_knowledge.observability.context import current_obs
from rfnry_knowledge.stores.document.base import BaseDocumentStore
from rfnry_knowledge.stores.vector.base import BaseVectorStore
from rfnry_knowledge.telemetry.context import current_query_row

logger = get_logger("retrieval.methods.keyword")

_GLOBAL_KEY = "__global__"

KeywordBackend = Literal["bm25", "postgres_fts"]


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


@dataclass
class _BM25Entry:
    index: BM25Okapi | None
    chunks: list[dict[str, Any]] = field(default_factory=list)
    last_used: float = 0.0


class KeywordRetrieval:
    """Lexical / exact-token keyword retrieval — one of the three pillars.

    Two backends:

    - ``backend="bm25"`` — in-memory BM25Okapi index built once per
      ``knowledge_id`` over the vector store's payloads. Cheap to set up,
      capped by ``bm25_max_chunks`` to avoid OOM. Requires ``vector_store``.
    - ``backend="postgres_fts"`` — Postgres full-text search + substring
      fallback against the document store. Persistent index lives in the
      database. Requires ``document_store``.
    """

    def __init__(
        self,
        backend: KeywordBackend = "bm25",
        vector_store: BaseVectorStore | None = None,
        document_store: BaseDocumentStore | None = None,
        bm25_max_indexes: int = 16,
        bm25_max_chunks: int = 50_000,
        bm25_tokenizer: Callable[[str], list[str]] | None = None,
        weight: float = 1.0,  # unbounded: caller-tuned RRF weight; downstream fusion handles outliers
        top_k: int | None = None,  # unbounded: soft override of RetrievalConfig.top_k
    ) -> None:
        if backend not in ("bm25", "postgres_fts"):
            raise ConfigurationError(f"KeywordRetrieval.backend must be 'bm25' or 'postgres_fts', got {backend!r}")
        if backend == "bm25":
            if vector_store is None:
                raise ConfigurationError("KeywordRetrieval(backend='bm25') requires vector_store")
            if not (1 <= bm25_max_indexes <= 1000):
                raise ConfigurationError(f"KeywordRetrieval.bm25_max_indexes={bm25_max_indexes} out of range [1, 1000]")
            if bm25_max_chunks > 200_000:
                raise ConfigurationError(
                    f"KeywordRetrieval.bm25_max_chunks={bm25_max_chunks} out of range [, 200_000] — "
                    "in-memory BM25 index at that size risks OOM; use a sparse-embeddings hybrid in "
                    "SemanticRetrieval instead"
                )
        else:
            if document_store is None:
                raise ConfigurationError("KeywordRetrieval(backend='postgres_fts') requires document_store")

        self._backend = backend
        self._vector_store = vector_store
        self._document_store = document_store
        self._bm25_max_indexes = bm25_max_indexes
        self._bm25_max_chunks = bm25_max_chunks
        self._tokenize_fn = bm25_tokenizer or _tokenize
        self._weight = weight
        self._top_k = top_k
        self._bm25_cache: dict[str, _BM25Entry] = {}
        self._bm25_lock = asyncio.Lock()

    def clone_for_store(
        self,
        store: BaseVectorStore | BaseDocumentStore,
    ) -> KeywordRetrieval:
        if self._backend == "bm25":
            return KeywordRetrieval(
                backend="bm25",
                vector_store=store,  # type: ignore[arg-type]
                bm25_max_indexes=self._bm25_max_indexes,
                bm25_max_chunks=self._bm25_max_chunks,
                bm25_tokenizer=self._tokenize_fn,
                weight=self._weight,
                top_k=self._top_k,
            )
        return KeywordRetrieval(
            backend="postgres_fts",
            document_store=store,  # type: ignore[arg-type]
            weight=self._weight,
            top_k=self._top_k,
        )

    @property
    def name(self) -> str:
        return "keyword"

    @property
    def backend(self) -> KeywordBackend:
        return self._backend

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
            if self._backend == "bm25":
                results = await self._bm25_search(query, top_k, knowledge_id)
            else:
                results = await self._postgres_fts_search(query, top_k, knowledge_id)
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            logger.info("%d results in %dms (backend=%s)", len(results), elapsed_ms, self._backend)
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
                        "backend": self._backend,
                        "chunks": len(results),
                        "duration_ms": elapsed_ms,
                    },
                )
            return results
        except Exception as exc:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            logger.warning("keyword retrieval (backend=%s) failed in %dms", self._backend, elapsed_ms, exc_info=True)
            if row is not None:
                row.method_errors += 1
                row.method_durations_ms[self.name] = elapsed_ms
            if obs is not None:
                await obs.emit(
                    "retrieval.method.error",
                    f"{self.name} retrieval failed",
                    level="error",
                    context={"method_name": self.name, "backend": self._backend, "duration_ms": elapsed_ms},
                    error=exc,
                )
            return []

    async def invalidate_cache(self, knowledge_id: str | None = None) -> None:
        if self._backend != "bm25":
            return
        key = knowledge_id if knowledge_id is not None else _GLOBAL_KEY
        async with self._bm25_lock:
            self._bm25_cache.pop(key, None)

    # ------------------------------------------------------------------
    # Postgres FTS backend
    # ------------------------------------------------------------------

    async def _postgres_fts_search(
        self,
        query: str,
        top_k: int,
        knowledge_id: str | None,
    ) -> list[RetrievedChunk]:
        assert self._document_store is not None
        matches = await self._document_store.search_content(query=query, knowledge_id=knowledge_id, top_k=top_k)
        return self._matches_to_chunks(matches)

    @staticmethod
    def _matches_to_chunks(matches: list[ContentMatch]) -> list[RetrievedChunk]:
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

    # ------------------------------------------------------------------
    # BM25 backend
    # ------------------------------------------------------------------

    async def _bm25_search(
        self,
        query: str,
        top_k: int,
        knowledge_id: str | None,
    ) -> list[RetrievedChunk]:
        key = knowledge_id if knowledge_id is not None else _GLOBAL_KEY
        async with self._bm25_lock:
            entry = self._bm25_cache.get(key)
        if entry is None:
            await self._build_bm25_index(knowledge_id)
            async with self._bm25_lock:
                entry = self._bm25_cache.get(key)
        if entry is None or entry.index is None or not entry.chunks:
            return []

        entry.last_used = time.monotonic()

        tokenized_query = self._tokenize_fn(query)
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

        logger.info("%d candidates from bm25 search", len(results))
        return results

    async def _build_bm25_index(self, knowledge_id: str | None) -> None:
        assert self._vector_store is not None
        key = knowledge_id if knowledge_id is not None else _GLOBAL_KEY

        async with self._bm25_lock:
            if key in self._bm25_cache:
                return

        filters = {"knowledge_id": knowledge_id} if knowledge_id is not None else None
        all_chunks: list[dict[str, Any]] = []
        offset = None
        capped = False

        while True:
            results, next_offset = await self._vector_store.scroll(filters=filters, limit=500, offset=offset)
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
                if len(all_chunks) >= self._bm25_max_chunks:
                    capped = True
                    break
            if capped or next_offset is None or not results:
                break
            offset = next_offset

        if capped:
            logger.warning(
                "bm25 index capped at %d chunks for knowledge_id=%s — "
                "consider a sparse-embeddings hybrid in SemanticRetrieval for larger corpora",
                self._bm25_max_chunks,
                knowledge_id,
            )

        loop = asyncio.get_running_loop()
        tokenize_fn = self._tokenize_fn
        if all_chunks:
            tokenized = await loop.run_in_executor(None, lambda: [tokenize_fn(c["content"]) for c in all_chunks])
            index: BM25Okapi | None = await loop.run_in_executor(None, lambda: BM25Okapi(tokenized))
        else:
            index = None

        async with self._bm25_lock:
            if key in self._bm25_cache:
                return
            self._evict_lru()
            self._bm25_cache[key] = _BM25Entry(index=index, chunks=all_chunks, last_used=time.monotonic())
        logger.info("built bm25 index for knowledge_id=%s: %d chunks", knowledge_id, len(all_chunks))

    def _evict_lru(self) -> None:
        if len(self._bm25_cache) < self._bm25_max_indexes:
            return
        oldest_key = min(self._bm25_cache, key=lambda k: self._bm25_cache[k].last_used)
        del self._bm25_cache[oldest_key]
        logger.info("evicted bm25 index for key=%s (lru)", oldest_key)
