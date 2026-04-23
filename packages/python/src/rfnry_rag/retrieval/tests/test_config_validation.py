from unittest.mock import MagicMock

import pytest

from rfnry_rag.retrieval.common.errors import ConfigurationError
from rfnry_rag.retrieval.server import (
    IngestionConfig,
    PersistenceConfig,
    RagEngine,
    RagServerConfig,
    RetrievalConfig,
)


def _mock_embeddings():
    m = MagicMock()
    m.model = "test"
    return m


def test_parent_chunk_size_must_be_nonnegative():
    with pytest.raises(ConfigurationError, match="non-negative"):
        IngestionConfig(embeddings=_mock_embeddings(), parent_chunk_size=-1)


def test_parent_chunk_size_must_exceed_chunk_size():
    with pytest.raises(ConfigurationError, match="greater than chunk_size"):
        IngestionConfig(embeddings=_mock_embeddings(), chunk_size=500, parent_chunk_size=300)


def test_valid_parent_chunk_config():
    config = IngestionConfig(embeddings=_mock_embeddings(), chunk_size=500, parent_chunk_size=1500)
    assert config.parent_chunk_size == 1500


def test_bm25_enabled_with_sparse_embeddings_raises():
    """bm25_enabled cannot coexist with sparse_embeddings — sparse supersedes BM25.
    The previous behavior was a warning + silent disable; that hid later
    misconfiguration when sparse_embeddings was removed without disabling bm25."""
    config = RagServerConfig(
        persistence=PersistenceConfig(vector_store=MagicMock()),
        ingestion=IngestionConfig(embeddings=_mock_embeddings(), sparse_embeddings=MagicMock()),
        retrieval=RetrievalConfig(bm25_enabled=True),
    )
    with pytest.raises(ConfigurationError, match="bm25_enabled.*sparse_embeddings"):
        RagEngine(config)._validate_config()


def test_bm25_enabled_without_sparse_is_fine():
    config = RagServerConfig(
        persistence=PersistenceConfig(vector_store=MagicMock()),
        ingestion=IngestionConfig(embeddings=_mock_embeddings()),
        retrieval=RetrievalConfig(bm25_enabled=True),
    )
    # Should not raise
    RagEngine(config)._validate_config()


def test_sparse_without_bm25_is_fine():
    config = RagServerConfig(
        persistence=PersistenceConfig(vector_store=MagicMock()),
        ingestion=IngestionConfig(embeddings=_mock_embeddings(), sparse_embeddings=MagicMock()),
        retrieval=RetrievalConfig(bm25_enabled=False),
    )
    RagEngine(config)._validate_config()
