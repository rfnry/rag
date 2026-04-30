"""IngestionConfig.analyze_concurrency feeds AnalyzedIngestionService's semaphore."""

import pytest

from rfnry_rag.common.errors import ConfigurationError


def test_config_default_is_5() -> None:
    from rfnry_rag.retrieval.server import IngestionConfig

    cfg = IngestionConfig()
    assert cfg.analyze_concurrency == 5


def test_config_bounds_out_of_range_raises() -> None:
    from rfnry_rag.retrieval.server import IngestionConfig

    with pytest.raises(ConfigurationError, match="analyze_concurrency"):
        IngestionConfig(analyze_concurrency=0)
    with pytest.raises(ConfigurationError, match="analyze_concurrency"):
        IngestionConfig(analyze_concurrency=101)
    with pytest.raises(ConfigurationError, match="analyze_concurrency"):
        IngestionConfig(analyze_concurrency=-1)


def test_config_accepts_boundary_values() -> None:
    from rfnry_rag.retrieval.server import IngestionConfig

    assert IngestionConfig(analyze_concurrency=1).analyze_concurrency == 1
    assert IngestionConfig(analyze_concurrency=100).analyze_concurrency == 100


def test_service_stores_configured_concurrency() -> None:
    """AnalyzedIngestionService.__init__ exposes _analyze_concurrency matching the arg."""
    from unittest.mock import AsyncMock

    from rfnry_rag.retrieval.modules.ingestion.analyze.service import AnalyzedIngestionService

    svc = AnalyzedIngestionService(
        embeddings=AsyncMock(),
        vector_store=AsyncMock(),
        metadata_store=AsyncMock(),
        embedding_model_name="fake",
        analyze_concurrency=12,
    )
    assert svc._analyze_concurrency == 12


def test_service_default_concurrency_is_5() -> None:
    from unittest.mock import AsyncMock

    from rfnry_rag.retrieval.modules.ingestion.analyze.service import AnalyzedIngestionService

    svc = AnalyzedIngestionService(
        embeddings=AsyncMock(),
        vector_store=AsyncMock(),
        metadata_store=AsyncMock(),
        embedding_model_name="fake",
    )
    assert svc._analyze_concurrency == 5
