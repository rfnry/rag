"""Graph store + ingestion LLM decoupling — a retrieval-only user with a
pre-populated graph shouldn't be forced to configure an ingestion LLM."""

from unittest.mock import MagicMock, patch

from rfnry_knowledge.config import IngestionConfig, KnowledgeEngineConfig, RetrievalConfig
from rfnry_knowledge.ingestion.methods.graph import GraphIngestion
from rfnry_knowledge.knowledge.engine import KnowledgeEngine
from rfnry_knowledge.retrieval.methods.graph import GraphRetrieval


def test_graph_store_without_ingestion_lm_client_does_not_raise() -> None:
    """A retrieval-only user with no LLM client must still pass validation."""
    graph_store = MagicMock()
    config = KnowledgeEngineConfig(
        ingestion=IngestionConfig(methods=[GraphIngestion(store=graph_store)]),
        retrieval=RetrievalConfig(methods=[GraphRetrieval(store=graph_store)]),
    )
    KnowledgeEngine(config)._validate_config()


def test_graph_store_with_lm_client_still_valid() -> None:
    graph_store = MagicMock()
    with patch("rfnry_knowledge.ingestion.methods.graph.build_registry", return_value=MagicMock()):
        config = KnowledgeEngineConfig(
            ingestion=IngestionConfig(
                methods=[GraphIngestion(store=graph_store, provider_client=MagicMock())],
            ),
            retrieval=RetrievalConfig(methods=[GraphRetrieval(store=graph_store)]),
        )
        KnowledgeEngine(config)._validate_config()
