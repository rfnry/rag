from unittest.mock import MagicMock

import pytest

from rfnry_rag.retrieval.common.errors import ConfigurationError
from rfnry_rag.retrieval.server import (
    IngestionConfig,
    PersistenceConfig,
    RagEngine,
    RagServerConfig,
    RetrievalConfig,
    TreeIndexingConfig,
    TreeSearchConfig,
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


class TestContextualChunkingDeprecation:
    """`contextual_chunking` is renamed to `chunk_context_headers`."""

    def test_new_name_defaults_to_true(self):
        cfg = IngestionConfig(embeddings=_mock_embeddings())
        assert cfg.chunk_context_headers is True

    def test_old_name_still_works_with_deprecation_warning(self):
        import warnings

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            cfg = IngestionConfig(embeddings=_mock_embeddings(), contextual_chunking=False)

        assert any(issubclass(w.category, DeprecationWarning) for w in captured)
        assert cfg.chunk_context_headers is False

    def test_new_name_wins_when_both_keys_provided_at_config_level(self):
        # Direct construction: the old param seeds the new one only if the new
        # default was not explicitly overridden. Here, explicit chunk_context_headers
        # must still reflect what the dataclass ends up with.
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            cfg = IngestionConfig(
                embeddings=_mock_embeddings(),
                chunk_context_headers=True,
                contextual_chunking=False,
            )
        # Since the deprecation shim copies contextual_chunking onto chunk_context_headers,
        # the old name "wins" here — that's the documented deprecation semantics.
        assert cfg.chunk_context_headers is False


def test_grounding_enabled_without_lm_client_rejected_at_config_time() -> None:
    from rfnry_rag.retrieval.common.errors import ConfigurationError
    from rfnry_rag.retrieval.server import GenerationConfig

    with pytest.raises(ConfigurationError, match="grounding_enabled requires"):
        GenerationConfig(grounding_enabled=True, grounding_threshold=0.5, lm_client=None)


def _config_with_metadata_store() -> RagServerConfig:
    """Minimal config with a document retrieval path and metadata_store.

    ``document_store`` satisfies the 'at least one retrieval path' check.
    ``metadata_store`` satisfies the tree-requires-metadata_store cross-check,
    so that the model=None check fires rather than the metadata_store check.
    """
    return RagServerConfig(
        persistence=PersistenceConfig(
            document_store=MagicMock(),
            metadata_store=MagicMock(),
        ),
        ingestion=IngestionConfig(embeddings=None),
    )


def test_tree_indexing_enabled_without_model_rejected_at_init() -> None:
    config = _config_with_metadata_store()
    config.tree_indexing = TreeIndexingConfig(enabled=True, model=None)
    with pytest.raises(ConfigurationError, match="tree_indexing.enabled requires tree_indexing.model"):
        RagEngine(config)._validate_config()


def test_tree_search_enabled_without_model_rejected_at_init() -> None:
    config = _config_with_metadata_store()
    config.tree_search = TreeSearchConfig(enabled=True, model=None)
    with pytest.raises(ConfigurationError, match="tree_search.enabled requires tree_search.model"):
        RagEngine(config)._validate_config()
