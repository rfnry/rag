from unittest.mock import AsyncMock

import pytest

from rfnry_rag.retrieval.common.models import SparseVector, VectorResult
from rfnry_rag.retrieval.modules.retrieval.methods.vector import VectorRetrieval


async def test_hybrid_search_called_when_sparse_available():
    vector_store = AsyncMock()
    vector_store.hybrid_search = AsyncMock(
        return_value=[
            VectorResult(
                point_id="p1",
                score=0.9,
                payload={
                    "content": "test",
                    "source_id": "s1",
                    "chunk_type": "child",
                    "parent_id": None,
                },
            ),
        ]
    )

    embeddings = AsyncMock()
    embeddings.embed = AsyncMock(return_value=[[0.1, 0.2]])

    sparse_embeddings = AsyncMock()
    sparse_embeddings.embed_sparse_query = AsyncMock(return_value=SparseVector(indices=[1], values=[0.8]))

    search = VectorRetrieval(
        vector_store=vector_store,
        embeddings=embeddings,
        sparse_embeddings=sparse_embeddings,
        parent_expansion=False,
        weight=1.0,
    )
    results = await search.search(query="test", top_k=5)

    vector_store.hybrid_search.assert_called_once()
    assert len(results) == 1


async def test_parent_expansion():
    parent_payload = {
        "content": "Full parent context with more detail.",
        "source_id": "s1",
        "chunk_type": "parent",
        "parent_id": "parent-1",
    }
    child_payload = {
        "content": "Short child match.",
        "source_id": "s1",
        "chunk_type": "child",
        "parent_id": "parent-1",
    }

    vector_store = AsyncMock()
    vector_store.search = AsyncMock(
        return_value=[
            VectorResult(point_id="c1", score=0.9, payload=child_payload),
        ]
    )
    vector_store.retrieve = AsyncMock(
        return_value=[
            VectorResult(point_id="parent-1", score=0.0, payload=parent_payload),
        ]
    )

    embeddings = AsyncMock()
    embeddings.embed = AsyncMock(return_value=[[0.1, 0.2]])

    search = VectorRetrieval(
        vector_store=vector_store,
        embeddings=embeddings,
        parent_expansion=True,
        weight=1.0,
    )
    results = await search.search(query="test", top_k=5)

    assert len(results) == 1
    assert "Full parent context" in results[0].content
    assert results[0].score == 0.9


async def test_dense_only_fallback():
    vector_store = AsyncMock()
    vector_store.search = AsyncMock(
        return_value=[
            VectorResult(
                point_id="p1",
                score=0.8,
                payload={
                    "content": "test",
                    "source_id": "s1",
                    "chunk_type": "child",
                    "parent_id": None,
                },
            ),
        ]
    )

    embeddings = AsyncMock()
    embeddings.embed = AsyncMock(return_value=[[0.1, 0.2]])

    search = VectorRetrieval(
        vector_store=vector_store,
        embeddings=embeddings,
        sparse_embeddings=None,
        parent_expansion=False,
        weight=1.0,
    )
    results = await search.search(query="test", top_k=5)

    vector_store.search.assert_called_once()
    assert len(results) == 1


async def test_parent_deduplication():
    """Multiple children sharing same parent should result in one expanded result."""
    vector_store = AsyncMock()
    vector_store.search = AsyncMock(
        return_value=[
            VectorResult(
                point_id="c1",
                score=0.9,
                payload={
                    "content": "Child 1",
                    "source_id": "s1",
                    "chunk_type": "child",
                    "parent_id": "parent-1",
                },
            ),
            VectorResult(
                point_id="c2",
                score=0.7,
                payload={
                    "content": "Child 2",
                    "source_id": "s1",
                    "chunk_type": "child",
                    "parent_id": "parent-1",
                },
            ),
        ]
    )
    vector_store.retrieve = AsyncMock(
        return_value=[
            VectorResult(
                point_id="parent-1",
                score=0.0,
                payload={
                    "content": "Full parent",
                    "source_id": "s1",
                    "chunk_type": "parent",
                    "parent_id": "parent-1",
                },
            ),
        ]
    )

    embeddings = AsyncMock()
    embeddings.embed = AsyncMock(return_value=[[0.1, 0.2]])

    search = VectorRetrieval(
        vector_store=vector_store,
        embeddings=embeddings,
        parent_expansion=True,
        weight=1.0,
    )
    results = await search.search(query="test", top_k=5)

    assert len(results) == 1
    # Score is now the sum of all child scores (0.9 + 0.7 = 1.6)
    assert results[0].score == pytest.approx(1.6)
