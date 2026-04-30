from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from rfnry_rag.retrieval.methods.vector import VectorRetrieval


@pytest.mark.asyncio
async def test_sparse_failure_falls_back_to_dense() -> None:
    """Sparse embedding failure inside VectorRetrieval._vector_search must not
    abort the search; it should degrade to dense-only."""
    store = SimpleNamespace(search=AsyncMock(return_value=[]), hybrid_search=AsyncMock(return_value=[]))
    embeddings = SimpleNamespace(embed=AsyncMock(return_value=[[0.1, 0.2]]))
    sparse = SimpleNamespace(embed_sparse_query=AsyncMock(side_effect=RuntimeError("sparse down")))
    retr = VectorRetrieval(
        store=store, embeddings=embeddings, sparse_embeddings=sparse, parent_expansion=False, bm25_enabled=False
    )
    result = await retr.search(query="q", top_k=3)
    assert result == []
    store.search.assert_awaited_once()
    store.hybrid_search.assert_not_called()


@pytest.mark.asyncio
async def test_dense_failure_returns_empty() -> None:
    """If dense embedding fails, fall through to the outer search() try/except
    (returns [] with warning log). Must not raise."""
    store = SimpleNamespace(search=AsyncMock(return_value=[]), hybrid_search=AsyncMock(return_value=[]))
    embeddings = SimpleNamespace(embed=AsyncMock(side_effect=RuntimeError("dense down")))
    sparse = SimpleNamespace(embed_sparse_query=AsyncMock(return_value=SimpleNamespace(indices=[1], values=[1.0])))
    retr = VectorRetrieval(
        store=store, embeddings=embeddings, sparse_embeddings=sparse, parent_expansion=False, bm25_enabled=False
    )
    result = await retr.search(query="q", top_k=3)
    assert result == []
