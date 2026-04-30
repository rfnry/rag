# src/rfnry-rag/retrieval/tests/test_vector_ingestion.py
from unittest.mock import AsyncMock, MagicMock

from rfnry_rag.ingestion.methods.vector import VectorIngestion
from rfnry_rag.retrieval.common.models import SparseVector


def _make_chunks(n=1):
    chunks = []
    for i in range(n):
        chunk = MagicMock()
        chunk.content = f"chunk {i}"
        chunk.embedding_text = f"chunk {i}"
        chunk.context = ""
        chunk.contextualized = ""
        chunk.page_number = 1
        chunk.section = None
        chunk.chunk_type = "child"
        chunk.parent_id = None
        chunks.append(chunk)
    return chunks


async def test_ingest_embeds_and_upserts():
    embeddings = AsyncMock()
    embeddings.embed = AsyncMock(return_value=[[0.1] * 128])
    vector_store = AsyncMock()
    vector_store.initialize = AsyncMock()
    vector_store.upsert = AsyncMock()

    method = VectorIngestion(
        vector_store=vector_store,
        embeddings=embeddings,
        embedding_model_name="test:model",
    )
    assert method.name == "vector"

    await method.ingest(
        source_id="src-1",
        knowledge_id="kb-1",
        source_type="manuals",
        source_weight=1.0,
        title="Test",
        full_text="chunk 0",
        chunks=_make_chunks(1),
        tags=[],
        metadata={},
    )
    embeddings.embed.assert_called_once()
    vector_store.upsert.assert_called_once()


async def test_ingest_with_sparse():
    embeddings = AsyncMock()
    embeddings.embed = AsyncMock(return_value=[[0.1] * 128])
    sparse = AsyncMock()
    sparse.embed_sparse = AsyncMock(return_value=[SparseVector(indices=[1], values=[0.8])])
    vector_store = AsyncMock()
    vector_store.initialize = AsyncMock()
    vector_store.upsert = AsyncMock()

    method = VectorIngestion(
        vector_store=vector_store,
        embeddings=embeddings,
        sparse_embeddings=sparse,
        embedding_model_name="test:model",
    )
    await method.ingest(
        source_id="src-1",
        knowledge_id=None,
        source_type=None,
        source_weight=1.0,
        title="Test",
        full_text="chunk 0",
        chunks=_make_chunks(1),
        tags=[],
        metadata={},
    )
    sparse.embed_sparse.assert_called_once()
    points = vector_store.upsert.call_args[0][0]
    assert points[0].sparse_vector is not None


async def test_delete():
    vector_store = AsyncMock()
    vector_store.delete = AsyncMock()

    method = VectorIngestion(
        vector_store=vector_store,
        embeddings=AsyncMock(),
        embedding_model_name="test:model",
    )
    await method.delete("src-1")
    vector_store.delete.assert_called_once_with({"source_id": "src-1"})
