"""Verify that BM25 index builds do not hold the cache lock across scroll I/O.

Concurrent builds for different knowledge_ids must overlap — if the lock were
held during scroll the two coroutines would serialize and max_concurrent would
stay at 1.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from rfnry_rag.retrieval.methods.vector import VectorRetrieval
from rfnry_rag.retrieval.search.keyword import KeywordSearch


async def test_bm25_build_does_not_hold_lock_across_scroll() -> None:
    """Concurrent BM25 builds for different knowledge_ids must overlap."""
    concurrent = 0
    max_concurrent = 0

    async def slow_scroll(filters=None, limit=500, offset=None):
        nonlocal concurrent, max_concurrent
        concurrent += 1
        max_concurrent = max(max_concurrent, concurrent)
        await asyncio.sleep(0.02)
        concurrent -= 1
        return [], None  # empty result terminates loop immediately

    store = MagicMock()
    store.scroll = AsyncMock(side_effect=slow_scroll)
    embeddings = MagicMock()
    method = VectorRetrieval(vector_store=store, embeddings=embeddings, bm25_enabled=True)

    await asyncio.gather(
        method._build_bm25_index(knowledge_id="kb-a"),
        method._build_bm25_index(knowledge_id="kb-b"),
    )

    assert max_concurrent >= 2, f"scroll ran serially under lock (max_concurrent={max_concurrent})"


async def test_keyword_search_build_does_not_hold_lock_across_scroll() -> None:
    """Concurrent KeywordSearch builds for different knowledge_ids must overlap."""
    concurrent = 0
    max_concurrent = 0

    async def slow_scroll(filters=None, limit=500, offset=None):
        nonlocal concurrent, max_concurrent
        concurrent += 1
        max_concurrent = max(max_concurrent, concurrent)
        await asyncio.sleep(0.02)
        concurrent -= 1
        return [], None  # empty result terminates loop immediately

    store = MagicMock()
    store.scroll = AsyncMock(side_effect=slow_scroll)
    searcher = KeywordSearch(vector_store=store)

    await asyncio.gather(
        searcher._build_index(knowledge_id="kb-a"),
        searcher._build_index(knowledge_id="kb-b"),
    )

    assert max_concurrent >= 2, f"scroll ran serially under lock (max_concurrent={max_concurrent})"
