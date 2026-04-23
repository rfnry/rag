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


@pytest.mark.parametrize("bad_dpi", [71, 601, 1_000, 10_000])
def test_ingestion_config_rejects_out_of_range_dpi(bad_dpi):
    with pytest.raises(ConfigurationError, match="dpi"):
        IngestionConfig(embeddings=_mock_embeddings(), dpi=bad_dpi)


@pytest.mark.parametrize("good_dpi", [72, 150, 300, 600])
def test_ingestion_config_accepts_in_range_dpi(good_dpi):
    IngestionConfig(embeddings=_mock_embeddings(), dpi=good_dpi)


@pytest.mark.parametrize("bad_k", [201, 1_000, 100_000])
def test_retrieval_config_rejects_huge_top_k(bad_k):
    with pytest.raises(ConfigurationError, match="top_k"):
        RetrievalConfig(top_k=bad_k)


def test_retrieval_config_rejects_huge_bm25_max_chunks():
    with pytest.raises(ConfigurationError, match="bm25_max_chunks"):
        RetrievalConfig(bm25_max_chunks=300_000)


def test_retrieval_config_accepts_sensible_bm25_max_chunks():
    cfg = RetrievalConfig(bm25_max_chunks=100_000)
    assert cfg.bm25_max_chunks == 100_000
