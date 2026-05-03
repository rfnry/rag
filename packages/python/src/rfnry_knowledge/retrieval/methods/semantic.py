from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from typing import Any

from rfnry_knowledge.common.logging import get_logger
from rfnry_knowledge.ingestion.embeddings.base import BaseEmbeddings
from rfnry_knowledge.ingestion.embeddings.batching import embed_batched
from rfnry_knowledge.ingestion.embeddings.sparse.base import BaseSparseEmbeddings
from rfnry_knowledge.models import RetrievedChunk, VectorResult
from rfnry_knowledge.observability.context import current_obs
from rfnry_knowledge.stores.vector.base import BaseVectorStore
from rfnry_knowledge.telemetry.context import current_query_row

logger = get_logger("retrieval.methods.semantic")


class SemanticRetrieval:
    """Dense embeddings + optional sparse hybrid search over the vector store."""

    def __init__(
        self,
        store: BaseVectorStore,
        embeddings: BaseEmbeddings,
        sparse_embeddings: BaseSparseEmbeddings | None = None,
        parent_expansion: bool = False,
        weight: float = 1.0,  # unbounded: caller-tuned RRF weight; downstream fusion handles outliers
        top_k: int | None = None,  # unbounded: soft override of RetrievalConfig.top_k
    ) -> None:
        self._store = store
        self._embeddings = embeddings
        self._sparse = sparse_embeddings
        self._parent_expansion = parent_expansion
        self._weight = weight
        self._top_k = top_k

    def clone_for_store(self, store: BaseVectorStore) -> SemanticRetrieval:
        return SemanticRetrieval(
            store=store,
            embeddings=self._embeddings,
            sparse_embeddings=self._sparse,
            parent_expansion=self._parent_expansion,
            weight=self._weight,
            top_k=self._top_k,
        )

    @property
    def name(self) -> str:
        return "semantic"

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
            results = await self._do_search(query, top_k, filters)
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
            logger.warning("semantic retrieval failed in %dms", elapsed_ms, exc_info=True)
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

    async def _do_search(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None,
    ) -> list[RetrievedChunk]:
        if self._sparse:
            dense_outcome, sparse_outcome = await asyncio.gather(
                embed_batched(self._embeddings, [query]),
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
            vectors = await embed_batched(self._embeddings, [query])
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
        parent_lookup: dict[str, dict],
    ) -> list[VectorResult]:
        groups: dict[str, list[VectorResult]] = defaultdict(list)
        for r in results:
            parent_id = r.payload.get("parent_id")
            if parent_id and parent_id in parent_lookup:
                groups[parent_id].append(r)

        merged: list[VectorResult] = []
        for parent_id, children in groups.items():
            parent_payload = dict(parent_lookup[parent_id])
            parent_payload["child_hit_count"] = len(children)
            parent_payload["expanded_from_children"] = [c.point_id for c in children]
            summed = sum(c.score for c in children)
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
