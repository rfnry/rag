from __future__ import annotations

import time
from typing import Any
from uuid import uuid4

from rfnry_rag.retrieval.common.logging import get_logger
from rfnry_rag.retrieval.common.models import SparseVector, VectorPoint
from rfnry_rag.retrieval.modules.ingestion.embeddings.base import BaseEmbeddings
from rfnry_rag.retrieval.modules.ingestion.embeddings.sparse.base import BaseSparseEmbeddings
from rfnry_rag.retrieval.modules.ingestion.embeddings.utils import embed_batched
from rfnry_rag.retrieval.modules.ingestion.models import ChunkedContent, ParsedPage
from rfnry_rag.retrieval.stores.vector.base import BaseVectorStore

logger = get_logger("ingestion.methods.vector")


class VectorIngestion:
    """Embed chunks and store as vector points."""

    def __init__(
        self,
        vector_store: BaseVectorStore,
        embeddings: BaseEmbeddings,
        embedding_model_name: str,
        sparse_embeddings: BaseSparseEmbeddings | None = None,
    ) -> None:
        self._store = vector_store
        self._embeddings = embeddings
        self._sparse = sparse_embeddings
        self._embedding_model_name = embedding_model_name

    @property
    def name(self) -> str:
        return "vector"

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
            texts = [c.embedding_text for c in chunks]
            vectors = await embed_batched(self._embeddings, texts)
            sparse_vectors = await self._embed_sparse_safe(texts)

            points = self._build_points(
                source_id,
                chunks,
                vectors,
                sparse_vectors,
                tags,
                metadata,
                knowledge_id,
                source_type,
                source_weight,
            )
            await self._store.upsert(points)

            elapsed = (time.perf_counter() - start) * 1000
            logger.info("%d chunks embedded in %.1fms", len(chunks), elapsed)
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000
            logger.warning("failed in %.1fms — %s", elapsed, exc)
            raise

    async def delete(self, source_id: str) -> None:
        await self._store.delete({"source_id": source_id})

    async def _embed_sparse_safe(self, texts: list[str]) -> list[SparseVector] | None:
        if not self._sparse:
            return None
        try:
            return await self._sparse.embed_sparse(texts)
        except Exception as exc:
            logger.warning("sparse embedding failed, continuing without: %s", exc)
            return None

    @staticmethod
    def _build_points(
        source_id: str,
        chunks: list[ChunkedContent],
        vectors: list[list[float]],
        sparse_vectors: list[SparseVector] | None,
        tags: list[str],
        metadata: dict[str, Any],
        knowledge_id: str | None,
        source_type: str | None,
        source_weight: float,
        chunk_index_offset: int = 0,
    ) -> list[VectorPoint]:
        points = []
        for idx, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True)):
            sparse = sparse_vectors[idx] if sparse_vectors else None
            point_id = chunk.parent_id if chunk.chunk_type == "parent" and chunk.parent_id else str(uuid4())
            points.append(
                VectorPoint(
                    point_id=point_id,
                    vector=vector,
                    sparse_vector=sparse,
                    payload={
                        "content": chunk.content,
                        "context": chunk.context,
                        "contextualized": chunk.contextualized,
                        "page_number": chunk.page_number,
                        "section": chunk.section,
                        "chunk_index": chunk_index_offset + idx,
                        "source_id": source_id,
                        "knowledge_id": knowledge_id,
                        "source_type": source_type,
                        "source_weight": source_weight,
                        "chunk_type": chunk.chunk_type,
                        "parent_id": chunk.parent_id,
                        "tags": tags,
                        "source_name": metadata.get("name", ""),
                        "file_url": metadata.get("file_url", ""),
                    },
                )
            )
        return points
