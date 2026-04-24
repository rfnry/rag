from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

from rfnry_rag.retrieval.modules.ingestion.models import ChunkedContent
from rfnry_rag.retrieval.modules.ingestion.methods.vector import VectorIngestion


def _make_chunk(text: str = "hello world") -> MagicMock:
    chunk = MagicMock()
    chunk.embedding_text = text
    chunk.content = text
    chunk.context = ""
    chunk.contextualized = ""
    chunk.page_number = 1
    chunk.section = None
    chunk.chunk_type = "child"
    chunk.parent_id = None
    return chunk


async def test_vector_ingestion_gathers_dense_and_sparse_embeddings() -> None:
    """Dense + sparse embeddings must be computed concurrently, not serially."""
    concurrent = 0
    max_concurrent = 0

    async def track_dense(*args, **kwargs):
        nonlocal concurrent, max_concurrent
        concurrent += 1
        max_concurrent = max(max_concurrent, concurrent)
        await asyncio.sleep(0.02)
        concurrent -= 1
        return [[0.1, 0.2, 0.3]]  # one dummy vector

    async def track_sparse(texts: list[str]):
        nonlocal concurrent, max_concurrent
        concurrent += 1
        max_concurrent = max(max_concurrent, concurrent)
        await asyncio.sleep(0.02)
        concurrent -= 1
        return [SimpleNamespace(indices=[0], values=[0.0])]

    # Dense embeddings mock — embed_batched is imported at module level so patch there.
    dense_mock = AsyncMock(side_effect=track_dense)

    # Sparse embeddings provider mock.
    sparse_provider = MagicMock()
    sparse_provider.embed_sparse = AsyncMock(side_effect=track_sparse)

    # Vector store mock.
    vector_store = MagicMock()
    vector_store.upsert = AsyncMock()

    # Dense embeddings provider (only used as an argument to embed_batched — the real
    # provider call is replaced by the patch below).
    embeddings_provider = MagicMock()

    vi = VectorIngestion(
        vector_store=vector_store,
        embeddings=embeddings_provider,
        embedding_model_name="test-model",
        sparse_embeddings=sparse_provider,
    )

    chunks = cast(list[ChunkedContent], [_make_chunk("chunk text")])

    with patch(
        "rfnry_rag.retrieval.modules.ingestion.methods.vector.embed_batched",
        new=dense_mock,
    ):
        await vi.ingest(
            source_id="src-1",
            knowledge_id="kb-1",
            source_type="document",
            source_weight=1.0,
            title="Test Doc",
            full_text="chunk text",
            chunks=chunks,
            tags=[],
            metadata={"name": "Test Doc"},
        )

    assert max_concurrent >= 2, (
        f"Expected dense and sparse to overlap (max_concurrent >= 2), got {max_concurrent}. "
        "They are still running serially."
    )
    dense_mock.assert_called_once()
    sparse_provider.embed_sparse.assert_called_once()
    vector_store.upsert.assert_called_once()


async def test_vector_ingestion_sparse_none_skips_gather() -> None:
    """When sparse_embeddings is None only dense is awaited (no gather)."""
    dense_mock = AsyncMock(return_value=[[0.1, 0.2, 0.3]])

    vector_store = MagicMock()
    vector_store.upsert = AsyncMock()

    vi = VectorIngestion(
        vector_store=vector_store,
        embeddings=MagicMock(),
        embedding_model_name="test-model",
        sparse_embeddings=None,
    )

    chunks = cast(list[ChunkedContent], [_make_chunk("chunk text")])

    with patch(
        "rfnry_rag.retrieval.modules.ingestion.methods.vector.embed_batched",
        new=dense_mock,
    ):
        await vi.ingest(
            source_id="src-2",
            knowledge_id=None,
            source_type=None,
            source_weight=1.0,
            title="Test",
            full_text="chunk text",
            chunks=chunks,
            tags=[],
            metadata={},
        )

    dense_mock.assert_called_once()
    vector_store.upsert.assert_called_once()
    # Sparse vectors should be None → no sparse field on the point.
    upserted_points = vector_store.upsert.call_args[0][0]
    assert all(p.sparse_vector is None for p in upserted_points)


async def test_vector_ingestion_sparse_failure_preserved() -> None:
    """If sparse embedding raises, _embed_sparse_safe swallows and returns None.
    With asyncio.gather the exception is swallowed *before* gather sees it, so
    the gather still resolves successfully and dense results are used."""
    dense_mock = AsyncMock(return_value=[[0.1, 0.2, 0.3]])

    failing_sparse = MagicMock()
    failing_sparse.embed_sparse = AsyncMock(side_effect=RuntimeError("sparse boom"))

    vector_store = MagicMock()
    vector_store.upsert = AsyncMock()

    vi = VectorIngestion(
        vector_store=vector_store,
        embeddings=MagicMock(),
        embedding_model_name="test-model",
        sparse_embeddings=failing_sparse,
    )

    chunks = cast(list[ChunkedContent], [_make_chunk("chunk text")])

    with patch(
        "rfnry_rag.retrieval.modules.ingestion.methods.vector.embed_batched",
        new=dense_mock,
    ):
        # Must not raise — _embed_sparse_safe already swallows the error.
        await vi.ingest(
            source_id="src-3",
            knowledge_id=None,
            source_type=None,
            source_weight=1.0,
            title="Test",
            full_text="chunk text",
            chunks=chunks,
            tags=[],
            metadata={},
        )

    vector_store.upsert.assert_called_once()
    upserted_points = vector_store.upsert.call_args[0][0]
    # sparse_vector should be None because _embed_sparse_safe returned None on failure.
    assert all(p.sparse_vector is None for p in upserted_points)
