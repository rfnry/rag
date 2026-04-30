"""Orchestrator for structured document retrieval."""

from typing import Any

from rfnry_rag.ingestion.embeddings.base import BaseEmbeddings
from rfnry_rag.logging import get_logger
from rfnry_rag.models import RetrievedChunk
from rfnry_rag.retrieval.enrich.enrichment import enrich_with_cross_references
from rfnry_rag.retrieval.enrich.field_search import results_to_chunks
from rfnry_rag.stores.vector.base import BaseVectorStore

logger = get_logger("enrich/retrieval")


class StructuredRetrievalService:
    def __init__(
        self,
        vector_store: BaseVectorStore,
        embeddings: BaseEmbeddings,
        top_k: int = 5,
        enrich_cross_references: bool = True,
    ) -> None:
        self._vector_store = vector_store
        self._embeddings = embeddings
        self._top_k = top_k
        self._enrich = enrich_cross_references

    async def retrieve(
        self,
        query: str,
        knowledge_id: str | None = None,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        if not query or not query.strip():
            return []

        top_k = top_k or self._top_k

        vectors = await self._embeddings.embed([query])
        semantic_filters: dict[str, Any] = {}
        if knowledge_id:
            semantic_filters["knowledge_id"] = knowledge_id

        semantic_results = await self._vector_store.search(
            vector=vectors[0], top_k=top_k * 2, filters=semantic_filters or None
        )
        semantic_chunks = results_to_chunks(semantic_results)

        merged = sorted(semantic_chunks, key=lambda x: x.score, reverse=True)[:top_k]
        logger.info("merged: %d results (semantic=%d)", len(merged), len(semantic_chunks))

        if self._enrich and merged:
            merged = await enrich_with_cross_references(merged, self._vector_store, knowledge_id)

        return merged
