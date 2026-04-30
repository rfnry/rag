from unittest.mock import MagicMock, patch

from rfnry_rag.ingestion.embeddings.sparse.fastembed import FastEmbedSparseEmbeddings
from rfnry_rag.retrieval.common.models import SparseVector


def test_sparse_vector_creation():
    sv = SparseVector(indices=[1, 5, 10], values=[0.8, 0.3, 0.1])
    assert len(sv.indices) == 3
    assert len(sv.values) == 3


async def test_fastembed_embed_sparse():
    mock_model = MagicMock()
    mock_result = MagicMock()
    mock_result.indices = [0, 5, 42]
    mock_result.values = [0.9, 0.4, 0.1]
    mock_model.embed.return_value = [mock_result]
    mock_model.model_name = "Qdrant/bm25"

    with patch(
        "fastembed.SparseTextEmbedding",
        return_value=mock_model,
    ):
        embeddings = FastEmbedSparseEmbeddings()
        result = await embeddings.embed_sparse(["test text"])

    assert len(result) == 1
    assert isinstance(result[0], SparseVector)
    assert result[0].indices == [0, 5, 42]
    assert result[0].values == [0.9, 0.4, 0.1]


async def test_fastembed_embed_sparse_query():
    mock_model = MagicMock()
    mock_result = MagicMock()
    mock_result.indices = [3, 7]
    mock_result.values = [0.6, 0.2]
    mock_model.query_embed.return_value = [mock_result]
    mock_model.model_name = "Qdrant/bm25"

    with patch(
        "fastembed.SparseTextEmbedding",
        return_value=mock_model,
    ):
        embeddings = FastEmbedSparseEmbeddings()
        result = await embeddings.embed_sparse_query("test query")

    assert isinstance(result, SparseVector)
    assert result.indices == [3, 7]
