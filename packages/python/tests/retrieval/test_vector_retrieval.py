"""Tests for SemanticRetrieval (dense + sparse hybrid). BM25 lives in KeywordRetrieval now."""

from unittest.mock import AsyncMock

from rfnry_knowledge.models import SparseVector, VectorResult
from rfnry_knowledge.retrieval.methods.semantic import SemanticRetrieval


async def test_dense_search():
    vector_store = AsyncMock()
    vector_store.search = AsyncMock(
        return_value=[
            VectorResult(
                point_id="p1",
                score=0.9,
                payload={"content": "test", "source_id": "s1", "chunk_type": "child", "parent_id": None},
            )
        ]
    )
    embeddings = AsyncMock()
    embeddings.embed = AsyncMock(return_value=[[0.1, 0.2]])
    method = SemanticRetrieval(store=vector_store, embeddings=embeddings, weight=1.0)
    assert method.name == "semantic"
    assert method.weight == 1.0
    results = await method.search(query="test", top_k=5)
    assert len(results) == 1
    vector_store.search.assert_called_once()


async def test_hybrid_search_with_sparse():
    vector_store = AsyncMock()
    vector_store.hybrid_search = AsyncMock(
        return_value=[
            VectorResult(
                point_id="p1",
                score=0.9,
                payload={"content": "test", "source_id": "s1", "chunk_type": "child", "parent_id": None},
            )
        ]
    )
    embeddings = AsyncMock()
    embeddings.embed = AsyncMock(return_value=[[0.1, 0.2]])
    sparse = AsyncMock()
    sparse.embed_sparse_query = AsyncMock(return_value=SparseVector(indices=[1], values=[0.8]))
    method = SemanticRetrieval(store=vector_store, embeddings=embeddings, sparse_embeddings=sparse, weight=1.5)
    results = await method.search(query="test", top_k=5)
    assert len(results) == 1
    vector_store.hybrid_search.assert_called_once()


async def test_error_returns_empty():
    vector_store = AsyncMock()
    vector_store.search = AsyncMock(side_effect=RuntimeError("connection lost"))
    embeddings = AsyncMock()
    embeddings.embed = AsyncMock(return_value=[[0.1, 0.2]])
    method = SemanticRetrieval(store=vector_store, embeddings=embeddings, weight=1.0)
    results = await method.search(query="test", top_k=5)
    assert results == []


async def test_name_and_weight_properties():
    method = SemanticRetrieval(store=AsyncMock(), embeddings=AsyncMock(), weight=2.5)
    assert method.name == "semantic"
    assert method.weight == 2.5
