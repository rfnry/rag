"""BM25 cap test — the scroll loop must stop at bm25_max_chunks instead of
loading an arbitrarily large corpus into RAM."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from rfnry_knowledge.retrieval.methods.keyword import KeywordRetrieval


def _result(point_id: str, content: str = "content") -> SimpleNamespace:
    return SimpleNamespace(
        point_id=point_id,
        payload={
            "content": content,
            "source_id": "s",
            "source_type": "manual",
            "source_weight": 1.0,
            "source_name": "n",
            "file_url": "u",
            "tags": [],
        },
    )


@pytest.mark.asyncio
async def test_bm25_scroll_stops_at_cap() -> None:
    page = [_result(f"p{i}") for i in range(500)]
    store = SimpleNamespace(scroll=AsyncMock(return_value=(page, "next")))
    retr = KeywordRetrieval(backend="bm25", vector_store=store, bm25_max_chunks=1000)
    await retr._build_bm25_index(knowledge_id=None)
    assert store.scroll.await_count == 2


@pytest.mark.asyncio
async def test_bm25_under_cap_reads_to_end() -> None:
    page = [_result(f"p{i}") for i in range(100)]
    store = SimpleNamespace(scroll=AsyncMock(side_effect=[(page, "next"), (page, None)]))
    retr = KeywordRetrieval(backend="bm25", vector_store=store, bm25_max_chunks=10000)
    await retr._build_bm25_index(knowledge_id=None)
    assert store.scroll.await_count == 2


def test_keyword_retrieval_default_bm25_max_chunks() -> None:
    """BM25 sizing knobs live on KeywordRetrieval, not RetrievalConfig."""
    method = KeywordRetrieval(backend="bm25", vector_store=SimpleNamespace())
    assert method._bm25_max_chunks == 50_000
