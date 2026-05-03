"""StructuredIngestion.analyze_concurrency feeds StructuredIngestionService's semaphore."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from rfnry_knowledge.exceptions import ConfigurationError
from rfnry_knowledge.ingestion.methods.structured import StructuredIngestion


def test_wrapper_default_is_5() -> None:
    method = StructuredIngestion(store=MagicMock(), embeddings=MagicMock())
    assert method._analyze_concurrency == 5


def test_wrapper_bounds_out_of_range_raises() -> None:
    with pytest.raises(ConfigurationError, match="analyze_concurrency"):
        StructuredIngestion(store=MagicMock(), embeddings=MagicMock(), analyze_concurrency=0)
    with pytest.raises(ConfigurationError, match="analyze_concurrency"):
        StructuredIngestion(store=MagicMock(), embeddings=MagicMock(), analyze_concurrency=101)
    with pytest.raises(ConfigurationError, match="analyze_concurrency"):
        StructuredIngestion(store=MagicMock(), embeddings=MagicMock(), analyze_concurrency=-1)


def test_wrapper_accepts_boundary_values() -> None:
    low = StructuredIngestion(store=MagicMock(), embeddings=MagicMock(), analyze_concurrency=1)
    high = StructuredIngestion(store=MagicMock(), embeddings=MagicMock(), analyze_concurrency=100)
    assert low._analyze_concurrency == 1
    assert high._analyze_concurrency == 100


def test_wrapper_text_skip_threshold_bounds() -> None:
    with pytest.raises(ConfigurationError, match="analyze_text_skip_threshold_chars"):
        StructuredIngestion(store=MagicMock(), embeddings=MagicMock(), analyze_text_skip_threshold_chars=-1)
    with pytest.raises(ConfigurationError, match="analyze_text_skip_threshold_chars"):
        StructuredIngestion(store=MagicMock(), embeddings=MagicMock(), analyze_text_skip_threshold_chars=100_001)


def test_wrapper_dpi_bounds_out_of_range_raises() -> None:
    with pytest.raises(ConfigurationError, match="dpi"):
        StructuredIngestion(store=MagicMock(), embeddings=MagicMock(), dpi=71)
    with pytest.raises(ConfigurationError, match="dpi"):
        StructuredIngestion(store=MagicMock(), embeddings=MagicMock(), dpi=601)


def test_wrapper_dpi_accepts_boundary_values() -> None:
    assert StructuredIngestion(store=MagicMock(), embeddings=MagicMock(), dpi=72)._dpi == 72
    assert StructuredIngestion(store=MagicMock(), embeddings=MagicMock(), dpi=600)._dpi == 600


def test_service_stores_configured_concurrency() -> None:
    """StructuredIngestionService.__init__ exposes _analyze_concurrency matching the arg."""
    from rfnry_knowledge.ingestion.structured.service import StructuredIngestionService

    svc = StructuredIngestionService(
        embeddings=AsyncMock(),
        vector_store=AsyncMock(),
        metadata_store=AsyncMock(),
        embedding_model_name="fake",
        analyze_concurrency=12,
    )
    assert svc._analyze_concurrency == 12


def test_service_default_concurrency_is_5() -> None:
    from rfnry_knowledge.ingestion.structured.service import StructuredIngestionService

    svc = StructuredIngestionService(
        embeddings=AsyncMock(),
        vector_store=AsyncMock(),
        metadata_store=AsyncMock(),
        embedding_model_name="fake",
    )
    assert svc._analyze_concurrency == 5
