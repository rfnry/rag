from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from rfnry_rag.models import RetrievedChunk
from rfnry_rag.retrieval.methods.vector import VectorRetrieval
from rfnry_rag.retrieval.search.service import RetrievalService


def _chunk(cid: str, score: float = 1.0) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=cid, source_id="s", content=cid, score=score)


@pytest.mark.asyncio
async def test_one_failing_query_does_not_crash_retrieval() -> None:
    """A failing query variant must not crash the whole retrieve() call — the
    surviving variants' results should still come through."""
    good_method = SimpleNamespace(
        name="good",
        weight=1.0,
        top_k=None,
        search=AsyncMock(return_value=[_chunk("c1")]),
    )
    rewriter = SimpleNamespace(rewrite=AsyncMock(return_value=["alt1", "alt2"]))

    service = RetrievalService(retrieval_methods=[good_method], top_k=5, query_rewriter=rewriter)

    call_counter = {"n": 0}

    async def flaky(
        q: str, *_a: Any, **_kw: Any
    ) -> tuple[list[list[RetrievedChunk]], list[float], dict[str, list[RetrievedChunk]] | None]:
        call_counter["n"] += 1
        if call_counter["n"] == 2:
            raise RuntimeError("boom")
        return ([[_chunk(f"c{call_counter['n']}")]], [1.0], None)

    service._search_single_query = flaky  # type: ignore[method-assign,assignment]

    results, _ = await service.retrieve(query="q")
    assert results
    assert {c.chunk_id for c in results} >= {"c1", "c3"}


@pytest.mark.asyncio
async def test_sparse_failure_falls_back_to_dense() -> None:
    """Sparse embedding failure inside VectorRetrieval._vector_search must not
    abort the search; it should degrade to dense-only."""
    store = SimpleNamespace(
        search=AsyncMock(return_value=[]),
        hybrid_search=AsyncMock(return_value=[]),
    )
    embeddings = SimpleNamespace(embed=AsyncMock(return_value=[[0.1, 0.2]]))
    sparse = SimpleNamespace(embed_sparse_query=AsyncMock(side_effect=RuntimeError("sparse down")))

    retr = VectorRetrieval(
        vector_store=store,  # type: ignore[arg-type]
        embeddings=embeddings,  # type: ignore[arg-type]
        sparse_embeddings=sparse,  # type: ignore[arg-type]
        parent_expansion=False,
        bm25_enabled=False,
    )

    result = await retr.search(query="q", top_k=3)
    assert result == []
    # Fallback path must have used dense search, not hybrid_search
    store.search.assert_awaited_once()
    store.hybrid_search.assert_not_called()


@pytest.mark.asyncio
async def test_dense_failure_returns_empty() -> None:
    """If dense embedding fails, fall through to the outer search() try/except
    (returns [] with warning log). Must not raise."""
    store = SimpleNamespace(
        search=AsyncMock(return_value=[]),
        hybrid_search=AsyncMock(return_value=[]),
    )
    embeddings = SimpleNamespace(embed=AsyncMock(side_effect=RuntimeError("dense down")))
    sparse = SimpleNamespace(embed_sparse_query=AsyncMock(return_value=SimpleNamespace(indices=[1], values=[1.0])))

    retr = VectorRetrieval(
        vector_store=store,  # type: ignore[arg-type]
        embeddings=embeddings,  # type: ignore[arg-type]
        sparse_embeddings=sparse,  # type: ignore[arg-type]
        parent_expansion=False,
        bm25_enabled=False,
    )

    result = await retr.search(query="q", top_k=3)
    assert result == []
