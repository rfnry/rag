"""Orchestrator for structured document retrieval."""

import asyncio
from typing import Any

from rfnry_rag.common.logging import get_logger
from rfnry_rag.ingestion.embeddings.base import BaseEmbeddings
from rfnry_rag.providers import LanguageModelClient, build_registry
from rfnry_rag.retrieval.common.models import RetrievedChunk
from rfnry_rag.retrieval.enrich.enrichment import enrich_with_cross_references
from rfnry_rag.retrieval.enrich.field_search import build_structured_filters, results_to_chunks
from rfnry_rag.retrieval.enrich.query_analyzer import analyze_query
from rfnry_rag.stores.vector.base import BaseVectorStore

logger = get_logger("enrich/retrieval")


class StructuredRetrievalService:
    def __init__(
        self,
        vector_store: BaseVectorStore,
        embeddings: BaseEmbeddings,
        lm_client: LanguageModelClient | None = None,
        top_k: int = 5,
        enrich_cross_references: bool = True,
    ) -> None:
        self._vector_store = vector_store
        self._embeddings = embeddings
        self._registry = build_registry(lm_client) if lm_client else None
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

        analysis = await self._analyze_query(query)
        logger.info(
            "query analysis: entities=%s",
            analysis["entity_references"],
        )

        vectors = await self._embeddings.embed([query])
        semantic_filters: dict[str, Any] = {}
        if knowledge_id:
            semantic_filters["knowledge_id"] = knowledge_id

        field_filters = None
        if analysis["entity_references"]:
            field_filters = build_structured_filters(analysis, knowledge_id)

        if field_filters:
            semantic_results, field_results = await asyncio.gather(
                self._vector_store.search(vector=vectors[0], top_k=top_k * 2, filters=semantic_filters or None),
                self._vector_store.search(vector=vectors[0], top_k=top_k, filters=field_filters),
            )
            field_chunks = results_to_chunks(field_results)
            for c in field_chunks:
                c.score = min(c.score * 1.2, 1.0)
        else:
            semantic_results = await self._vector_store.search(
                vector=vectors[0], top_k=top_k * 2, filters=semantic_filters or None
            )
            field_chunks = []

        semantic_chunks = results_to_chunks(semantic_results)

        merged = self._merge_results(semantic_chunks, field_chunks, top_k)
        logger.info(
            "merged: %d results (semantic=%d, field=%d)",
            len(merged),
            len(semantic_chunks),
            len(field_chunks),
        )

        if self._enrich and merged:
            merged = await enrich_with_cross_references(merged, self._vector_store, knowledge_id)

        return merged

    async def _analyze_query(self, query: str) -> dict[str, Any]:
        """Use BAML to extract structured search terms from natural language."""
        if not self._registry:
            return {"entity_references": [], "keywords": [], "intent": ""}
        return await analyze_query(query, self._registry)

    @staticmethod
    def _merge_results(
        semantic: list[RetrievedChunk],
        field: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Merge semantic and field results, deduplicating by chunk_id."""
        seen: set[str] = set()
        merged: list[RetrievedChunk] = []

        for c in field:
            if c.chunk_id not in seen:
                seen.add(c.chunk_id)
                merged.append(c)

        for c in semantic:
            if c.chunk_id not in seen:
                seen.add(c.chunk_id)
                merged.append(c)

        merged.sort(key=lambda x: x.score, reverse=True)
        return merged[:top_k]
