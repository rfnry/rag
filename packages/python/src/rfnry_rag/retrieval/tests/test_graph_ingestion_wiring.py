"""Wiring: IngestionConfig.graph flows through to the mapper via both services."""
from __future__ import annotations

from types import SimpleNamespace

from rfnry_rag.retrieval.modules.ingestion.graph.config import GraphIngestionConfig


def test_analyzed_service_stores_graph_config() -> None:
    from rfnry_rag.retrieval.modules.ingestion.analyze.service import AnalyzedIngestionService

    cfg = GraphIngestionConfig(unclassified_relation_default="MENTIONS")
    svc = AnalyzedIngestionService(
        embeddings=SimpleNamespace(),
        vector_store=SimpleNamespace(),
        metadata_store=SimpleNamespace(),
        embedding_model_name="e",
        graph_config=cfg,
    )
    assert svc._graph_config is cfg


def test_graph_ingestion_method_stores_graph_config() -> None:
    from rfnry_rag.retrieval.modules.ingestion.methods.graph import GraphIngestion

    cfg = GraphIngestionConfig()
    svc = GraphIngestion(
        graph_store=SimpleNamespace(),
        lm_client=None,
        graph_config=cfg,
    )
    assert svc._graph_config is cfg


def test_analyzed_service_defaults_graph_config_to_agnostic_empty() -> None:
    """When no config is passed, the service builds a default empty one
    (agnostic — category-fallback type inference + MENTIONS xref default)."""
    from rfnry_rag.retrieval.modules.ingestion.analyze.service import AnalyzedIngestionService

    svc = AnalyzedIngestionService(
        embeddings=SimpleNamespace(),
        vector_store=SimpleNamespace(),
        metadata_store=SimpleNamespace(),
        embedding_model_name="e",
    )
    assert isinstance(svc._graph_config, GraphIngestionConfig)
    assert svc._graph_config.entity_type_patterns == []
    assert svc._graph_config.unclassified_relation_default == "MENTIONS"


def test_graph_ingestion_method_defaults_to_agnostic_empty() -> None:
    from rfnry_rag.retrieval.modules.ingestion.methods.graph import GraphIngestion

    svc = GraphIngestion(graph_store=SimpleNamespace(), lm_client=None)
    assert isinstance(svc._graph_config, GraphIngestionConfig)
    assert svc._graph_config.entity_type_patterns == []
    assert svc._graph_config.unclassified_relation_default == "MENTIONS"
