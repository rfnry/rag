from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from rank_bm25 import BM25Okapi

from rfnry_rag.exceptions import ConfigurationError
from rfnry_rag.ingestion.embeddings.base import BaseEmbeddings
from rfnry_rag.ingestion.embeddings.sparse.base import BaseSparseEmbeddings
from rfnry_rag.logging import get_logger
from rfnry_rag.models import RetrievedChunk, VectorResult
from rfnry_rag.observability.context import current_obs
from rfnry_rag.retrieval.search.fusion import reciprocal_rank_fusion
from rfnry_rag.stores.vector.base import BaseVectorStore
from rfnry_rag.telemetry.context import current_query_row

logger = get_logger("retrieval.methods.vector")

_GLOBAL_KEY = "__global__"


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


@dataclass
class _BM25Entry:
    index: BM25Okapi | None
    chunks: list[dict[str, Any]] = field(default_factory=list)
    last_used: float = 0.0


class VectorRetrieval:
    """Unified vector retrieval method combining dense/hybrid search with optional BM25."""

    def __init__(
        self,
        store: BaseVectorStore,
        embeddings: BaseEmbeddings,
        sparse_embeddings: BaseSparseEmbeddings | None = None,
        parent_expansion: bool = False,
        bm25_enabled: bool = False,
        bm25_max_indexes: int = 16,
        bm25_max_chunks: int = 50_000,
        bm25_tokenizer: Callable[[str], list[str]] | None = None,
        weight: float = 1.0,  # unbounded: caller-tuned RRF weight; downstream fusion handles outliers
        top_k: int | None = None,  # unbounded: soft override of RetrievalConfig.top_k
    ) -> None:
        if not (1 <= bm25_max_indexes <= 1000):
            raise ConfigurationError(f"VectorRetrieval.bm25_max_indexes={bm25_max_indexes} out of range [1, 1000]")
        if bm25_max_chunks > 200_000:
            raise ConfigurationError(
                f"VectorRetrieval.bm25_max_chunks={bm25_max_chunks} out of range [, 200_000] — "
                "in-memory BM25 index at that size risks OOM; use sparse_embeddings instead"
            )
        self._store = store
        self._embeddings = embeddings
        self._sparse = sparse_embeddings
        self._parent_expansion = parent_expansion
        self._bm25_enabled = bm25_enabled
        self._bm25_max_indexes = bm25_max_indexes
        self._bm25_max_chunks = bm25_max_chunks
        self._tokenize_fn = bm25_tokenizer or _tokenize
        self._weight = weight
        self._top_k = top_k
        self._bm25_cache: dict[str, _BM25Entry] = {}
        self._bm25_lock = asyncio.Lock()

    def clone_for_store(self, store: BaseVectorStore) -> VectorRetrieval:
        clone = VectorRetrieval(
            store=store,
            embeddings=self._embeddings,
            sparse_embeddings=self._sparse,
            parent_expansion=self._parent_expansion,
            bm25_enabled=self._bm25_enabled,
            bm25_max_indexes=self._bm25_max_indexes,
            bm25_max_chunks=self._bm25_max_chunks,
            bm25_tokenizer=self._tokenize_fn,
            weight=self._weight,
            top_k=self._top_k,
        )
        return clone

    @property
    def name(self) -> str:
        return "vector"

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
            results = await self._do_search(query, top_k, filters, knowledge_id)
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            logger.info("%d results in %dms", len(results), elapsed_ms)
            if row is not None:
                row.method_durations_ms[self.name] = elapsed_ms
                if self.name not in row.methods_used:
                    row.methods_used.append(self.name)
                row.chunks_retrieved += len(results)
            if obs is not None:
                await obs.emit(
                    "info",
                    "retrieval.method.success",
                    f"{self.name} retrieval ok",
                    method_name=self.name,
                    chunks=len(results),
                    duration_ms=elapsed_ms,
                )
            return results
        except Exception as exc:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            logger.warning("vector retrieval failed in %dms", elapsed_ms, exc_info=True)
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

    async def invalidate_cache(self, knowledge_id: str | None = None) -> None:
        key = knowledge_id if knowledge_id is not None else _GLOBAL_KEY
        async with self._bm25_lock:
            self._bm25_cache.pop(key, None)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _do_search(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None,
        knowledge_id: str | None,
    ) -> list[RetrievedChunk]:
        # Dense / hybrid vector search
        dense_results = await self._vector_search(query, top_k, filters)

        if not self._bm25_enabled:
            return dense_results

        # BM25 keyword search
        bm25_results = await self._bm25_search(query, top_k, knowledge_id)

        if not bm25_results:
            return dense_results

        return reciprocal_rank_fusion([dense_results, bm25_results])[:top_k]

    # ------------------------------------------------------------------
    # Dense / hybrid search
    # ------------------------------------------------------------------

    async def _vector_search(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None,
    ) -> list[RetrievedChunk]:
        if self._sparse:
            dense_outcome, sparse_outcome = await asyncio.gather(
                self._embeddings.embed([query]),
                self._sparse.embed_sparse_query(query),
                return_exceptions=True,
            )
            if isinstance(dense_outcome, BaseException):
                logger.warning("dense embedding failed: %s", dense_outcome)
                return []
            if isinstance(sparse_outcome, BaseException):
                logger.warning("sparse embedding failed: %s — falling back to dense only", sparse_outcome)
                sparse_vector = None
            else:
                sparse_vector = sparse_outcome
            query_vector = dense_outcome[0] if dense_outcome else None
            if not query_vector:
                logger.warning("embedding returned no vectors for query")
                return []
            if sparse_vector is not None:
                results = await self._store.hybrid_search(
                    vector=query_vector,
                    sparse_vector=sparse_vector,
                    top_k=top_k,
                    filters=filters,
                )
                logger.info("%d candidates from hybrid search", len(results))
            else:
                results = await self._store.search(vector=query_vector, top_k=top_k, filters=filters)
                logger.info("%d candidates from dense fallback (sparse failed)", len(results))
        else:
            vectors = await self._embeddings.embed([query])
            if not vectors:
                logger.warning("embedding returned no vectors for query")
                return []
            query_vector = vectors[0]
            results = await self._store.search(vector=query_vector, top_k=top_k, filters=filters)
            logger.info("%d candidates from dense search", len(results))

        results = [r for r in results if r.payload.get("chunk_type", "child") == "child"]

        if self._parent_expansion and results:
            results = await self._expand_parents(results)

        return [self._result_to_chunk(r) for r in results]

    async def _expand_parents(self, results: list[VectorResult]) -> list[VectorResult]:
        """For each child result with a parent_id, fetch the parent and return its content instead.

        Multiple children sharing the same parent are collapsed into one result whose score
        is the sum of all child scores (stronger multi-hit signal). Results without a parent_id
        (already non-child chunks) are preserved as-is.
        """
        children = [r for r in results if r.payload.get("parent_id")]
        non_children = [r for r in results if not r.payload.get("parent_id")]

        if not children:
            return results

        parent_ids = {r.payload["parent_id"] for r in children}
        parents = await self._store.retrieve(list(parent_ids))
        parent_lookup = {p.point_id: p.payload for p in parents}

        merged = self._merge_children_into_parents(children, parent_lookup)
        return non_children + merged

    @staticmethod
    def _merge_children_into_parents(
        results: list[VectorResult],
        parent_lookup: dict[str, dict],  # {parent_id: parent_payload}
    ) -> list[VectorResult]:
        """Collapse child results sharing a parent into one result per parent.

        Score is the sum of child scores. child_hit_count is added to the
        payload so downstream reranker/LLM can observe multi-hit strength.
        Children whose parent_id is not in parent_lookup are dropped.
        """
        from collections import defaultdict

        groups: dict[str, list[VectorResult]] = defaultdict(list)
        for r in results:
            parent_id = r.payload.get("parent_id")
            if parent_id and parent_id in parent_lookup:
                groups[parent_id].append(r)
            # else: child without a resolvable parent — drop silently

        merged: list[VectorResult] = []
        for parent_id, children in groups.items():
            parent_payload = dict(parent_lookup[parent_id])  # copy
            parent_payload["child_hit_count"] = len(children)
            parent_payload["expanded_from_children"] = [c.point_id for c in children]
            summed = sum(c.score for c in children)
            # Use the first child's point_id as the stable result id to preserve
            # downstream deduplication semantics; content comes from parent.
            first = children[0]
            merged.append(
                VectorResult(
                    point_id=first.point_id,
                    score=summed,
                    payload=parent_payload,
                )
            )

        return merged

    @staticmethod
    def _result_to_chunk(r: VectorResult) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=r.point_id,
            content=r.payload.get("content", ""),
            score=r.score,
            page_number=r.payload.get("page_number"),
            section=r.payload.get("section"),
            source_id=r.payload.get("source_id", ""),
            source_type=r.payload.get("source_type"),
            source_weight=r.payload.get("source_weight", 1.0),
            source_metadata={
                "name": r.payload.get("source_name", ""),
                "file_url": r.payload.get("file_url", ""),
                "tags": r.payload.get("tags", []),
                "chunk_type": r.payload.get("chunk_type", "child"),
                "parent_id": r.payload.get("parent_id"),
            },
        )

    # ------------------------------------------------------------------
    # BM25 keyword search (from KeywordSearch)
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
        key = knowledge_id if knowledge_id is not None else _GLOBAL_KEY

        # Fast path: cache hit, no work.
        async with self._bm25_lock:
            if key in self._bm25_cache:
                return

        # Collect chunks OUTSIDE the lock. Multiple builds for different
        # keys can proceed concurrently; the vector store is the natural
        # serialization point.
        filters = {"knowledge_id": knowledge_id} if knowledge_id is not None else None
        all_chunks: list[dict[str, Any]] = []
        offset = None
        capped = False

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
                if len(all_chunks) >= self._bm25_max_chunks:
                    capped = True
                    break
            if capped or next_offset is None or not results:
                break
            offset = next_offset

        if capped:
            logger.warning(
                "bm25 index capped at %d chunks for knowledge_id=%s — consider sparse_embeddings for larger corpora",
                self._bm25_max_chunks,
                knowledge_id,
            )

        # Tokenize + build in thread pool — pure-Python CPU work.
        loop = asyncio.get_running_loop()
        tokenize_fn = self._tokenize_fn
        if all_chunks:
            tokenized = await loop.run_in_executor(None, lambda: [tokenize_fn(c["content"]) for c in all_chunks])
            index: BM25Okapi | None = await loop.run_in_executor(None, lambda: BM25Okapi(tokenized))
        else:
            index = None

        # Re-check-then-write under the lock. If another build landed first,
        # discard ours.
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
