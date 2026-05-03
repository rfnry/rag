"""Cross-reference enrichment for structured retrieval results."""

import asyncio
from typing import Any

from rfnry_knowledge.common.logging import get_logger
from rfnry_knowledge.models import RetrievedChunk
from rfnry_knowledge.stores.vector.base import BaseVectorStore

logger = get_logger("enrich/retrieval/enrichment")


async def enrich_with_cross_references(
    chunks: list[RetrievedChunk],
    vector_store: BaseVectorStore,
    knowledge_id: str | None = None,
    max_enrichments: int = 3,
) -> list[RetrievedChunk]:
    """For each retrieved chunk, fetch cross-referenced pages as additional context.

    Returns the original chunks plus any linked pages not already in the result set.
    """
    if not chunks:
        return chunks

    existing_pages = {c.page_number for c in chunks if c.page_number is not None}
    enriched: list[RetrievedChunk] = list(chunks)
    pages_to_fetch: set[int] = set()

    for chunk in chunks:
        cross_refs = chunk.metadata.get("cross_references", [])
        for ref in cross_refs[:max_enrichments]:
            if isinstance(ref, int):
                target: int | None = ref
            elif isinstance(ref, dict):
                raw_target = ref.get("target_page")
                target = raw_target if isinstance(raw_target, int) else None
            else:
                continue
            if target is not None and target not in existing_pages and target not in pages_to_fetch:
                pages_to_fetch.add(target)

    if not pages_to_fetch:
        return enriched

    from rfnry_knowledge.retrieval.enrich.field_search import results_to_chunks

    async def _fetch_page(page_num: int) -> tuple[int, list]:
        filters: dict[str, Any] = {"page_number": page_num}
        if knowledge_id:
            filters["knowledge_id"] = knowledge_id
        results, _ = await vector_store.scroll(filters=filters, limit=1)
        return page_num, results

    fetched = await asyncio.gather(*[_fetch_page(pn) for pn in pages_to_fetch])
    for page_num, results in fetched:
        if results:
            linked = results_to_chunks(results)
            if linked:
                linked[0].score = 0.0
                enriched.append(linked[0])
                existing_pages.add(page_num)

    if len(enriched) > len(chunks):
        logger.info("enriched with %d cross-referenced pages", len(enriched) - len(chunks))

    return enriched
