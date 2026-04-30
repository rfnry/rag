"""Graph store + ingestion LLM decoupling — a retrieval-only user with a
pre-populated graph shouldn't be forced to configure an ingestion LLM."""

from unittest.mock import MagicMock, patch

from rfnry_rag.config import IngestionConfig, RagEngineConfig, RetrievalConfig
from rfnry_rag.ingestion.methods.graph import GraphIngestion
from rfnry_rag.retrieval.methods.graph import GraphRetrieval
from rfnry_rag.server import RagEngine


def test_graph_store_without_ingestion_lm_client_does_not_raise() -> None:
    """A retrieval-only user with no LLM client must still pass validation."""
    graph_store = MagicMock()
    config = RagEngineConfig(
        ingestion=IngestionConfig(methods=[GraphIngestion(store=graph_store)]),
        retrieval=RetrievalConfig(methods=[GraphRetrieval(store=graph_store)]),
    )
    RagEngine(config)._validate_config()


def test_graph_store_with_lm_client_still_valid() -> None:
    graph_store = MagicMock()
    with patch("rfnry_rag.ingestion.methods.graph.build_registry", return_value=MagicMock()):
        config = RagEngineConfig(
            ingestion=IngestionConfig(
                methods=[GraphIngestion(store=graph_store, lm_client=MagicMock())],
            ),
            retrieval=RetrievalConfig(methods=[GraphRetrieval(store=graph_store)]),
        )
        RagEngine(config)._validate_config()
