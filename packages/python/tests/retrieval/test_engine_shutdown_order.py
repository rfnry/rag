"""Regression: shutdown() must tear down stores in reverse-init order.

Init order:  metadata -> document -> graph -> vector
Shutdown order (correct): vector -> graph -> document -> metadata
"""

from unittest.mock import AsyncMock, MagicMock

from rfnry_knowledge.config import IngestionConfig, KnowledgeEngineConfig, RetrievalConfig
from rfnry_knowledge.ingestion.methods.entity import EntityIngestion
from rfnry_knowledge.ingestion.methods.keyword import KeywordIngestion
from rfnry_knowledge.ingestion.methods.semantic import SemanticIngestion
from rfnry_knowledge.knowledge.engine import KnowledgeEngine
from rfnry_knowledge.retrieval.methods.entity import EntityRetrieval
from rfnry_knowledge.retrieval.methods.keyword import KeywordRetrieval
from rfnry_knowledge.retrieval.methods.semantic import SemanticRetrieval


async def test_shutdown_tears_down_in_reverse_init_order() -> None:
    calls: list[str] = []

    metadata = MagicMock()
    metadata.initialize = AsyncMock()
    metadata.list_sources = AsyncMock(return_value=[])
    metadata.shutdown = AsyncMock(side_effect=lambda: calls.append("metadata"))

    document = MagicMock()
    document.initialize = AsyncMock()
    document.shutdown = AsyncMock(side_effect=lambda: calls.append("document"))

    graph = MagicMock()
    graph.initialize = AsyncMock()
    graph.shutdown = AsyncMock(side_effect=lambda: calls.append("graph"))

    vector = MagicMock()
    vector.initialize = AsyncMock()
    vector.shutdown = AsyncMock(side_effect=lambda: calls.append("vector"))
    vector.collections = ["knowledge"]

    embeddings = MagicMock()
    embeddings.model = "test"
    embeddings.embedding_dimension = AsyncMock(return_value=128)

    cfg = KnowledgeEngineConfig(
        metadata_store=metadata,
        ingestion=IngestionConfig(
            methods=[
                SemanticIngestion(store=vector, embeddings=embeddings),
                KeywordIngestion(store=document),
                EntityIngestion(store=graph),
            ],
        ),
        retrieval=RetrievalConfig(
            methods=[
                SemanticRetrieval(store=vector, embeddings=embeddings),
                KeywordRetrieval(backend="postgres_fts", document_store=document),
                EntityRetrieval(store=graph),
            ],
        ),
    )
    engine = KnowledgeEngine(cfg)
    await engine.initialize()
    await engine.shutdown()

    assert calls == ["vector", "graph", "document", "metadata"]


async def test_shutdown_clears_service_refs() -> None:
    document = MagicMock()
    document.initialize = AsyncMock()
    document.shutdown = AsyncMock()

    cfg = KnowledgeEngineConfig(
        ingestion=IngestionConfig(methods=[KeywordIngestion(store=document)]),
        retrieval=RetrievalConfig(methods=[KeywordRetrieval(backend="postgres_fts", document_store=document)]),
    )
    engine = KnowledgeEngine(cfg)
    await engine.initialize()
    await engine.shutdown()

    assert engine._ingestion_service is None
    assert engine._retrieval_service is None
    assert engine._retrieval_namespace is None
    assert engine._ingestion_namespace is None
    assert engine._retrieval_by_collection == {}
    assert engine._ingestion_by_collection == {}


async def test_shutdown_is_idempotent() -> None:
    """Calling shutdown() twice must not re-invoke store shutdowns."""
    vector = MagicMock()
    vector.initialize = AsyncMock()
    vector.shutdown = AsyncMock()
    vector.collections = ["knowledge"]

    metadata = MagicMock()
    metadata.initialize = AsyncMock()
    metadata.list_sources = AsyncMock(return_value=[])
    metadata.shutdown = AsyncMock()

    embeddings = MagicMock()
    embeddings.model = "test"
    embeddings.embedding_dimension = AsyncMock(return_value=128)

    cfg = KnowledgeEngineConfig(
        metadata_store=metadata,
        ingestion=IngestionConfig(
            methods=[SemanticIngestion(store=vector, embeddings=embeddings)],
        ),
        retrieval=RetrievalConfig(
            methods=[SemanticRetrieval(store=vector, embeddings=embeddings)],
        ),
    )
    engine = KnowledgeEngine(cfg)
    await engine.initialize()

    await engine.shutdown()
    await engine.shutdown()

    assert vector.shutdown.await_count == 1
    assert metadata.shutdown.await_count == 1
