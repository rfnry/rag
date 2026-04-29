"""Tests for tree service wiring in RagEngine."""

from rfnry_rag.retrieval.server import RagServerConfig, TreeIndexingConfig, TreeSearchConfig


class TestRagServerConfigIncludesTree:
    def test_config_has_tree_indexing_field(self) -> None:
        """RagServerConfig exposes a tree_indexing field of type TreeIndexingConfig."""
        from unittest.mock import MagicMock

        vector_store = MagicMock()
        embeddings = MagicMock()
        embeddings.model = "test"

        from rfnry_rag.retrieval.server import IngestionConfig, PersistenceConfig

        cfg = RagServerConfig(
            persistence=PersistenceConfig(vector_store=vector_store),
            ingestion=IngestionConfig(embeddings=embeddings),
        )
        assert isinstance(cfg.tree_indexing, TreeIndexingConfig)

    def test_config_has_tree_search_field(self) -> None:
        """RagServerConfig exposes a tree_search field of type TreeSearchConfig."""
        from unittest.mock import MagicMock

        vector_store = MagicMock()
        embeddings = MagicMock()
        embeddings.model = "test"

        from rfnry_rag.retrieval.server import IngestionConfig, PersistenceConfig

        cfg = RagServerConfig(
            persistence=PersistenceConfig(vector_store=vector_store),
            ingestion=IngestionConfig(embeddings=embeddings),
        )
        assert isinstance(cfg.tree_search, TreeSearchConfig)

    def test_config_accepts_custom_tree_indexing(self) -> None:
        """RagServerConfig accepts a custom TreeIndexingConfig."""
        from unittest.mock import MagicMock

        vector_store = MagicMock()
        embeddings = MagicMock()
        embeddings.model = "test"

        from rfnry_rag.retrieval.server import IngestionConfig, PersistenceConfig

        tree_cfg = TreeIndexingConfig(enabled=True, toc_scan_pages=10)
        cfg = RagServerConfig(
            persistence=PersistenceConfig(vector_store=vector_store),
            ingestion=IngestionConfig(embeddings=embeddings),
            tree_indexing=tree_cfg,
        )
        assert cfg.tree_indexing.enabled is True
        assert cfg.tree_indexing.toc_scan_pages == 10

    def test_config_accepts_custom_tree_search(self) -> None:
        """RagServerConfig accepts a custom TreeSearchConfig."""
        from unittest.mock import MagicMock

        vector_store = MagicMock()
        embeddings = MagicMock()
        embeddings.model = "test"

        from rfnry_rag.retrieval.server import IngestionConfig, PersistenceConfig

        tree_cfg = TreeSearchConfig(enabled=True, max_steps=10)
        cfg = RagServerConfig(
            persistence=PersistenceConfig(vector_store=vector_store),
            ingestion=IngestionConfig(embeddings=embeddings),
            tree_search=tree_cfg,
        )
        assert cfg.tree_search.enabled is True
        assert cfg.tree_search.max_steps == 10


class TestRagServerConfigTreeDefaultsDisabled:
    def test_tree_indexing_defaults_disabled(self) -> None:
        """TreeIndexingConfig defaults to enabled=False."""
        cfg = TreeIndexingConfig()
        assert cfg.enabled is False

    def test_tree_search_defaults_disabled(self) -> None:
        """TreeSearchConfig defaults to enabled=False."""
        cfg = TreeSearchConfig()
        assert cfg.enabled is False

    def test_server_config_tree_indexing_disabled_by_default(self) -> None:
        """RagServerConfig.tree_indexing.enabled is False by default."""
        from unittest.mock import MagicMock

        vector_store = MagicMock()
        embeddings = MagicMock()
        embeddings.model = "test"

        from rfnry_rag.retrieval.server import IngestionConfig, PersistenceConfig

        cfg = RagServerConfig(
            persistence=PersistenceConfig(vector_store=vector_store),
            ingestion=IngestionConfig(embeddings=embeddings),
        )
        assert cfg.tree_indexing.enabled is False

    def test_server_config_tree_search_disabled_by_default(self) -> None:
        """RagServerConfig.tree_search.enabled is False by default."""
        from unittest.mock import MagicMock

        vector_store = MagicMock()
        embeddings = MagicMock()
        embeddings.model = "test"

        from rfnry_rag.retrieval.server import IngestionConfig, PersistenceConfig

        cfg = RagServerConfig(
            persistence=PersistenceConfig(vector_store=vector_store),
            ingestion=IngestionConfig(embeddings=embeddings),
        )
        assert cfg.tree_search.enabled is False
