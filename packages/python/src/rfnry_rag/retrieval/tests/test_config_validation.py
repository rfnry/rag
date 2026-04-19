from unittest.mock import MagicMock

import pytest

from rfnry_rag.retrieval.common.errors import ConfigurationError
from rfnry_rag.retrieval.server import IngestionConfig


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
