import re
from unittest.mock import AsyncMock

import pytest

from rfnry_rag.exceptions import ConfigurationError
from rfnry_rag.models import SparseVector, VectorResult
from rfnry_rag.retrieval.methods.vector import VectorRetrieval


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
    method = VectorRetrieval(store=vector_store, embeddings=embeddings, weight=1.0)
    assert method.name == "vector"
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
    method = VectorRetrieval(store=vector_store, embeddings=embeddings, sparse_embeddings=sparse, weight=1.5)
    results = await method.search(query="test", top_k=5)
    assert len(results) == 1
    vector_store.hybrid_search.assert_called_once()


async def test_bm25_enabled_fuses_results():
    vector_store = AsyncMock()
    vector_store.search = AsyncMock(
        return_value=[
            VectorResult(
                point_id="p1",
                score=0.9,
                payload={"content": "matching content", "source_id": "s1", "chunk_type": "child", "parent_id": None},
            )
        ]
    )
    vector_store.scroll = AsyncMock(
        return_value=(
            [
                VectorResult(
                    point_id="p1",
                    score=0.0,
                    payload={
                        "content": "matching content",
                        "source_id": "s1",
                        "chunk_type": "child",
                        "source_type": None,
                        "source_weight": 1.0,
                        "source_name": "",
                        "file_url": "",
                        "tags": [],
                        "page_number": None,
                        "section": None,
                    },
                )
            ],
            None,
        )
    )
    embeddings = AsyncMock()
    embeddings.embed = AsyncMock(return_value=[[0.1, 0.2]])
    method = VectorRetrieval(
        store=vector_store, embeddings=embeddings, bm25_enabled=True, bm25_max_indexes=16, weight=1.0
    )
    results = await method.search(query="matching content", top_k=5)
    assert len(results) >= 1


async def test_error_returns_empty():
    vector_store = AsyncMock()
    vector_store.search = AsyncMock(side_effect=RuntimeError("connection lost"))
    embeddings = AsyncMock()
    embeddings.embed = AsyncMock(return_value=[[0.1, 0.2]])
    method = VectorRetrieval(store=vector_store, embeddings=embeddings, weight=1.0)
    results = await method.search(query="test", top_k=5)
    assert results == []


async def test_name_and_weight_properties():
    method = VectorRetrieval(store=AsyncMock(), embeddings=AsyncMock(), weight=2.5)
    assert method.name == "vector"
    assert method.weight == 2.5


async def test_invalidate_cache():
    """BM25 cache should be cleared after invalidation."""
    vector_store = AsyncMock()
    vector_store.search = AsyncMock(
        return_value=[
            VectorResult(
                point_id="p1",
                score=0.9,
                payload={"content": "cached content", "source_id": "s1", "chunk_type": "child", "parent_id": None},
            )
        ]
    )
    vector_store.scroll = AsyncMock(
        return_value=(
            [
                VectorResult(
                    point_id="p1",
                    score=0.0,
                    payload={
                        "content": "cached content",
                        "source_id": "s1",
                        "chunk_type": "child",
                        "source_type": None,
                        "source_weight": 1.0,
                        "source_name": "",
                        "file_url": "",
                        "tags": [],
                        "page_number": None,
                        "section": None,
                    },
                )
            ],
            None,
        )
    )
    embeddings = AsyncMock()
    embeddings.embed = AsyncMock(return_value=[[0.1, 0.2]])
    method = VectorRetrieval(store=vector_store, embeddings=embeddings, bm25_enabled=True, weight=1.0)
    await method.search(query="cached content", top_k=5)
    assert vector_store.scroll.call_count == 1
    await method.invalidate_cache(knowledge_id=None)
    await method.search(query="cached content", top_k=5)
    assert vector_store.scroll.call_count == 2


async def test_custom_bm25_tokenizer():
    """Custom tokenizer should be used for BM25 indexing and query."""
    call_log = []

    def custom_tokenizer(text: str) -> list[str]:
        tokens = re.findall("\\w+(?:[-.]\\w+)*", text.lower())
        call_log.append(text)
        return tokens

    vector_store = AsyncMock()
    vector_store.search = AsyncMock(
        return_value=[
            VectorResult(
                point_id="p1",
                score=0.9,
                payload={
                    "content": "part 1756-EN2T specs",
                    "source_id": "s1",
                    "chunk_type": "child",
                    "parent_id": None,
                },
            )
        ]
    )
    vector_store.scroll = AsyncMock(
        return_value=(
            [
                VectorResult(
                    point_id="p1",
                    score=0.0,
                    payload={
                        "content": "part 1756-EN2T specs",
                        "source_id": "s1",
                        "chunk_type": "child",
                        "source_type": None,
                        "source_weight": 1.0,
                        "source_name": "",
                        "file_url": "",
                        "tags": [],
                        "page_number": None,
                        "section": None,
                    },
                )
            ],
            None,
        )
    )
    embeddings = AsyncMock()
    embeddings.embed = AsyncMock(return_value=[[0.1, 0.2]])
    method = VectorRetrieval(
        store=vector_store, embeddings=embeddings, bm25_enabled=True, bm25_tokenizer=custom_tokenizer, weight=1.0
    )
    await method.search(query="1756-EN2T", top_k=5)
    assert len(call_log) > 0


def test_bm25_max_indexes_bounds() -> None:
    store = AsyncMock()
    embeddings = AsyncMock()
    with pytest.raises(ConfigurationError, match="bm25_max_indexes"):
        VectorRetrieval(store=store, embeddings=embeddings, bm25_max_indexes=0)
    with pytest.raises(ConfigurationError, match="bm25_max_indexes"):
        VectorRetrieval(store=store, embeddings=embeddings, bm25_max_indexes=1001)


def test_bm25_max_chunks_bounds() -> None:
    store = AsyncMock()
    embeddings = AsyncMock()
    with pytest.raises(ConfigurationError, match="bm25_max_chunks"):
        VectorRetrieval(store=store, embeddings=embeddings, bm25_max_chunks=200_001)
