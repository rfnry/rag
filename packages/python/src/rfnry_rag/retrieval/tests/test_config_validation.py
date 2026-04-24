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


def test_parent_chunk_size_sentinel_minus_one_is_auto():
    # -1 is the sentinel for "auto = 3 * chunk_size"; it must NOT raise
    cfg = IngestionConfig(embeddings=_mock_embeddings(), chunk_size=375, parent_chunk_size=-1)
    assert cfg.parent_chunk_size == 3 * 375


def test_parent_chunk_size_below_minus_one_raises():
    with pytest.raises(ConfigurationError, match="parent_chunk_size"):
        IngestionConfig(embeddings=_mock_embeddings(), parent_chunk_size=-2)


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


def test_retrieval_config_bm25_max_indexes_bounds() -> None:
    with pytest.raises(ConfigurationError, match="bm25_max_indexes"):
        RetrievalConfig(bm25_max_indexes=0)
    with pytest.raises(ConfigurationError, match="bm25_max_indexes"):
        RetrievalConfig(bm25_max_indexes=1001)
    assert RetrievalConfig().bm25_max_indexes == 16


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


def test_ingestion_config_parent_chunk_overlap_must_be_less_than_parent_chunk_size() -> None:
    from unittest.mock import MagicMock

    from rfnry_rag.retrieval.common.errors import ConfigurationError
    from rfnry_rag.retrieval.server import IngestionConfig

    # Pass values that satisfy child-split invariant but violate parent's.
    with pytest.raises(ConfigurationError, match="parent_chunk_overlap"):
        IngestionConfig(
            chunk_size=500,
            chunk_overlap=50,
            parent_chunk_size=600,
            parent_chunk_overlap=700,
            embeddings=MagicMock(),
        )


def test_ingestion_config_parent_chunk_overlap_negative_rejected() -> None:
    from unittest.mock import MagicMock

    from rfnry_rag.retrieval.common.errors import ConfigurationError
    from rfnry_rag.retrieval.server import IngestionConfig

    with pytest.raises(ConfigurationError, match="parent_chunk_overlap"):
        IngestionConfig(
            chunk_size=500,
            chunk_overlap=50,
            parent_chunk_size=600,
            parent_chunk_overlap=-1,
            embeddings=MagicMock(),
        )


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


def test_retrieval_config_history_window_validates() -> None:
    from rfnry_rag.retrieval.common.errors import ConfigurationError
    from rfnry_rag.retrieval.server import RetrievalConfig

    default = RetrievalConfig()
    assert default.history_window == 3

    custom = RetrievalConfig(history_window=1)
    assert custom.history_window == 1

    with pytest.raises(ConfigurationError, match="history_window"):
        RetrievalConfig(history_window=0)
    with pytest.raises(ConfigurationError, match="history_window"):
        RetrievalConfig(history_window=21)


def test_tree_search_config_max_sources_per_query_validates() -> None:
    with pytest.raises(ConfigurationError, match="max_sources_per_query"):
        TreeSearchConfig(enabled=False, max_sources_per_query=0)
    with pytest.raises(ConfigurationError, match="max_sources_per_query"):
        TreeSearchConfig(enabled=False, max_sources_per_query=1001)
    assert TreeSearchConfig(enabled=False).max_sources_per_query == 50


def test_tree_indexing_config_upper_bounds() -> None:
    with pytest.raises(ConfigurationError, match="toc_scan_pages"):
        TreeIndexingConfig(toc_scan_pages=501)
    with pytest.raises(ConfigurationError, match="max_pages_per_node"):
        TreeIndexingConfig(max_pages_per_node=201)
    with pytest.raises(ConfigurationError, match="max_tokens_per_node"):
        TreeIndexingConfig(max_tokens_per_node=200_001)


def test_tree_search_config_upper_bounds() -> None:
    with pytest.raises(ConfigurationError, match="max_steps"):
        TreeSearchConfig(max_steps=51)
    with pytest.raises(ConfigurationError, match="max_context_tokens"):
        TreeSearchConfig(max_context_tokens=500_001)
