"""Exact field matching on Qdrant payload metadata for structured documents."""

from typing import Any

from rfnry_rag.retrieval.common.logging import get_logger
from rfnry_rag.retrieval.common.models import RetrievedChunk, VectorResult

logger = get_logger("enrich/retrieval/field")


def build_structured_filters(
    analysis: dict[str, Any],
    knowledge_id: str | None = None,
) -> dict[str, Any]:
    """Build Qdrant filter dict from query analysis."""
    filters: dict[str, Any] = {}
    if knowledge_id:
        filters["knowledge_id"] = knowledge_id
    if analysis.get("entity_references"):
        filters["entities"] = analysis["entity_references"]
    return filters


def results_to_chunks(results: list[VectorResult]) -> list[RetrievedChunk]:
    """Convert raw Qdrant results to RetrievedChunk objects."""
    chunks = []
    for r in results:
        p = r.payload
        chunks.append(
            RetrievedChunk(
                chunk_id=r.point_id,
                source_id=p.get("source_id", ""),
                content=p.get("content", ""),
                score=r.score,
                page_number=p.get("page_number"),
                section=p.get("section"),
                source_type=p.get("source_type"),
                source_weight=p.get("source_weight", 1.0),
                metadata={
                    "page_type": p.get("page_type", ""),
                    "entities": p.get("entities", []),
                    "cross_references": p.get("cross_references", []),
                },
                source_metadata={
                    "name": p.get("source_name", ""),
                    "file_url": p.get("file_url", ""),
                },
            )
        )
    return chunks
