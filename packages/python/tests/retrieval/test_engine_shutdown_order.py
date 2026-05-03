"""Regression: shutdown() must tear down stores in reverse-init order.

Init order:  metadata -> document -> graph -> vector
Shutdown order (correct): vector -> graph -> document -> metadata
"""

from unittest.mock import AsyncMock, MagicMock

from rfnry_knowledge.config import IngestionConfig, KnowledgeEngineConfig, RetrievalConfig
from rfnry_knowledge.ingestion.methods.document import DocumentIngestion
from rfnry_knowledge.ingestion.methods.graph import GraphIngestion
from rfnry_knowledge.ingestion.methods.vector import VectorIngestion
from rfnry_knowledge.knowledge.engine import KnowledgeEngine
from rfnry_knowledge.retrieval.methods.document import DocumentRetrieval
from rfnry_knowledge.retrieval.methods.graph import GraphRetrieval
from rfnry_knowledge.retrieval.methods.vector import VectorRetrieval


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
                VectorIngestion(store=vector, embeddings=embeddings),
                DocumentIngestion(store=document),
                GraphIngestion(store=graph),
            ],
        ),
        retrieval=RetrievalConfig(
            methods=[
                VectorRetrieval(store=vector, embeddings=embeddings),
                DocumentRetrieval(store=document),
                GraphRetrieval(store=graph),
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
        ingestion=IngestionConfig(methods=[DocumentIngestion(store=document)]),
        retrieval=RetrievalConfig(methods=[DocumentRetrieval(store=document)]),
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
            methods=[VectorIngestion(store=vector, embeddings=embeddings)],
        ),
        retrieval=RetrievalConfig(
            methods=[VectorRetrieval(store=vector, embeddings=embeddings)],
        ),
    )
    engine = KnowledgeEngine(cfg)
    await engine.initialize()

    await engine.shutdown()
    await engine.shutdown()

    assert vector.shutdown.await_count == 1
    assert metadata.shutdown.await_count == 1
