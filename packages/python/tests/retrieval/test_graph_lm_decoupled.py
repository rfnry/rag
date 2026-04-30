"""Graph store + ingestion LLM decoupling — a retrieval-only user with a
pre-populated graph shouldn't be forced to configure an ingestion LLM."""

from unittest.mock import MagicMock

from rfnry_rag.server import (
    IngestionConfig,
    PersistenceConfig,
    RagEngine,
    RagEngineConfig,
)


def test_graph_store_without_ingestion_lm_client_does_not_raise() -> None:
    """Previously this raised ConfigurationError at engine-init. Now it logs
    a warning and lets retrieval-only workflows proceed."""
    config = RagEngineConfig(
        persistence=PersistenceConfig(graph_store=MagicMock()),
        ingestion=IngestionConfig(),  # no lm_client
    )
    # Must not raise
    RagEngine(config)._validate_config()


def test_graph_store_with_lm_client_still_valid() -> None:
    config = RagEngineConfig(
        persistence=PersistenceConfig(graph_store=MagicMock()),
        ingestion=IngestionConfig(lm_client=MagicMock()),
    )
    RagEngine(config)._validate_config()
