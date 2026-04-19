import asyncio
from typing import Any

from rfnry_rag.retrieval.common.logging import get_logger
from rfnry_rag.retrieval.common.models import RetrievedChunk, VectorResult
from rfnry_rag.retrieval.modules.ingestion.embeddings.base import BaseEmbeddings
from rfnry_rag.retrieval.modules.ingestion.embeddings.sparse.base import BaseSparseEmbeddings
from rfnry_rag.retrieval.stores.vector.base import BaseVectorStore

logger = get_logger("search/retrieval/vector")


class VectorSearch:
    def __init__(
        self,
        vector_store: BaseVectorStore,
        embeddings: BaseEmbeddings,
        sparse_embeddings: BaseSparseEmbeddings | None = None,
        parent_expansion: bool = False,
    ) -> None:
        self._store = vector_store
        self._embeddings = embeddings
        self._sparse = sparse_embeddings
        self._parent_expansion = parent_expansion

    async def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        if self._sparse:
            dense_result, sparse_vector = await asyncio.gather(
                self._embeddings.embed([query]),
                self._sparse.embed_sparse_query(query),
            )
            query_vector = dense_result[0] if dense_result else None
            if not query_vector:
                logger.warning("embedding returned no vectors for query")
                return []
            results = await self._store.hybrid_search(
                vector=query_vector,
                sparse_vector=sparse_vector,
                top_k=top_k,
                filters=filters,
            )
            logger.info("%d candidates from hybrid search", len(results))
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
        """For each child result with a parent_id, fetch the parent and return its content instead."""
        parent_ids = set()
        for r in results:
            pid = r.payload.get("parent_id")
            if pid:
                parent_ids.add(pid)

        if not parent_ids:
            return results

        parents = await self._store.retrieve(list(parent_ids))
        parent_map = {p.point_id: p for p in parents}

        expanded = []
        seen_parents: set[str] = set()

        for r in results:
            pid = r.payload.get("parent_id")
            if pid and pid in parent_map:
                if pid in seen_parents:
                    continue
                seen_parents.add(pid)
                parent = parent_map[pid]
                expanded.append(
                    VectorResult(
                        point_id=r.point_id,
                        score=r.score,
                        payload={**parent.payload, "expanded_from_child": r.point_id},
                    )
                )
            else:
                expanded.append(r)

        return expanded

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
