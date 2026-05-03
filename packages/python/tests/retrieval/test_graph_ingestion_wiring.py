"""Wiring: graph_config flows through to the mapper via both services."""

from __future__ import annotations

from types import SimpleNamespace

from rfnry_knowledge.config.entity import EntityIngestionConfig


def test_analyzed_service_stores_graph_config() -> None:
    from rfnry_knowledge.ingestion.analyze.service import AnalyzedIngestionService

    cfg = EntityIngestionConfig(unclassified_relation_default="MENTIONS")
    svc = AnalyzedIngestionService(
        embeddings=SimpleNamespace(),
        vector_store=SimpleNamespace(),
        metadata_store=SimpleNamespace(),
        embedding_model_name="e",
        graph_config=cfg,
    )
    assert svc._graph_config is cfg


def test_graph_ingestion_method_stores_graph_config() -> None:
    from rfnry_knowledge.ingestion.methods.entity import EntityIngestion

    cfg = EntityIngestionConfig()
    svc = EntityIngestion(store=SimpleNamespace(), provider_client=None, graph_config=cfg)
    assert svc._graph_config is cfg


def test_analyzed_service_defaults_graph_config_to_agnostic_empty() -> None:
    """When no config is passed, the service builds a default empty one
    (agnostic — category-fallback type inference + MENTIONS xref default)."""
    from rfnry_knowledge.ingestion.analyze.service import AnalyzedIngestionService

    svc = AnalyzedIngestionService(
        embeddings=SimpleNamespace(),
        vector_store=SimpleNamespace(),
        metadata_store=SimpleNamespace(),
        embedding_model_name="e",
    )
    assert isinstance(svc._graph_config, EntityIngestionConfig)
    assert svc._graph_config.entity_type_patterns == []
    assert svc._graph_config.unclassified_relation_default == "MENTIONS"


def test_graph_ingestion_method_defaults_to_agnostic_empty() -> None:
    from rfnry_knowledge.ingestion.methods.entity import EntityIngestion

    svc = EntityIngestion(store=SimpleNamespace(), provider_client=None)
    assert isinstance(svc._graph_config, EntityIngestionConfig)
    assert svc._graph_config.entity_type_patterns == []
    assert svc._graph_config.unclassified_relation_default == "MENTIONS"
