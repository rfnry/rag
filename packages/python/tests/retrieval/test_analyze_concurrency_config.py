"""AnalyzedIngestion.analyze_concurrency feeds AnalyzedIngestionService's semaphore."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from rfnry_rag.exceptions import ConfigurationError
from rfnry_rag.ingestion.methods.analyzed import AnalyzedIngestion


def test_wrapper_default_is_5() -> None:
    method = AnalyzedIngestion(store=MagicMock(), embeddings=MagicMock())
    assert method._analyze_concurrency == 5


def test_wrapper_bounds_out_of_range_raises() -> None:
    with pytest.raises(ConfigurationError, match="analyze_concurrency"):
        AnalyzedIngestion(store=MagicMock(), embeddings=MagicMock(), analyze_concurrency=0)
    with pytest.raises(ConfigurationError, match="analyze_concurrency"):
        AnalyzedIngestion(store=MagicMock(), embeddings=MagicMock(), analyze_concurrency=101)
    with pytest.raises(ConfigurationError, match="analyze_concurrency"):
        AnalyzedIngestion(store=MagicMock(), embeddings=MagicMock(), analyze_concurrency=-1)


def test_wrapper_accepts_boundary_values() -> None:
    assert AnalyzedIngestion(store=MagicMock(), embeddings=MagicMock(), analyze_concurrency=1)._analyze_concurrency == 1
    assert (
        AnalyzedIngestion(store=MagicMock(), embeddings=MagicMock(), analyze_concurrency=100)._analyze_concurrency
        == 100
    )


def test_wrapper_text_skip_threshold_bounds() -> None:
    with pytest.raises(ConfigurationError, match="analyze_text_skip_threshold_chars"):
        AnalyzedIngestion(store=MagicMock(), embeddings=MagicMock(), analyze_text_skip_threshold_chars=-1)
    with pytest.raises(ConfigurationError, match="analyze_text_skip_threshold_chars"):
        AnalyzedIngestion(store=MagicMock(), embeddings=MagicMock(), analyze_text_skip_threshold_chars=100_001)


def test_wrapper_dpi_bounds_out_of_range_raises() -> None:
    with pytest.raises(ConfigurationError, match="dpi"):
        AnalyzedIngestion(store=MagicMock(), embeddings=MagicMock(), dpi=71)
    with pytest.raises(ConfigurationError, match="dpi"):
        AnalyzedIngestion(store=MagicMock(), embeddings=MagicMock(), dpi=601)


def test_wrapper_dpi_accepts_boundary_values() -> None:
    assert AnalyzedIngestion(store=MagicMock(), embeddings=MagicMock(), dpi=72)._dpi == 72
    assert AnalyzedIngestion(store=MagicMock(), embeddings=MagicMock(), dpi=600)._dpi == 600


def test_service_stores_configured_concurrency() -> None:
    """AnalyzedIngestionService.__init__ exposes _analyze_concurrency matching the arg."""
    from rfnry_rag.ingestion.analyze.service import AnalyzedIngestionService

    svc = AnalyzedIngestionService(
        embeddings=AsyncMock(),
        vector_store=AsyncMock(),
        metadata_store=AsyncMock(),
        embedding_model_name="fake",
        analyze_concurrency=12,
    )
    assert svc._analyze_concurrency == 12


def test_service_default_concurrency_is_5() -> None:
    from rfnry_rag.ingestion.analyze.service import AnalyzedIngestionService

    svc = AnalyzedIngestionService(
        embeddings=AsyncMock(),
        vector_store=AsyncMock(),
        metadata_store=AsyncMock(),
        embedding_model_name="fake",
    )
    assert svc._analyze_concurrency == 5
