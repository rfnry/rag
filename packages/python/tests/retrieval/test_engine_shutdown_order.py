"""Regression: shutdown() must tear down stores in reverse-init order.

Init order:  metadata -> document -> graph -> vector
Shutdown order (correct): vector -> graph -> document -> metadata
"""

from unittest.mock import AsyncMock, MagicMock

from rfnry_rag.config import IngestionConfig, RagEngineConfig, RetrievalConfig
from rfnry_rag.ingestion.methods.document import DocumentIngestion
from rfnry_rag.ingestion.methods.graph import GraphIngestion
from rfnry_rag.ingestion.methods.vector import VectorIngestion
from rfnry_rag.retrieval.methods.document import DocumentRetrieval
from rfnry_rag.retrieval.methods.graph import GraphRetrieval
from rfnry_rag.retrieval.methods.vector import VectorRetrieval
from rfnry_rag.server import RagEngine


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

    cfg = RagEngineConfig(
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
    engine = RagEngine(cfg)
    await engine.initialize()
    await engine.shutdown()

    assert calls == ["vector", "graph", "document", "metadata"]


async def test_shutdown_clears_service_refs() -> None:
    document = MagicMock()
    document.initialize = AsyncMock()
    document.shutdown = AsyncMock()

    cfg = RagEngineConfig(
        ingestion=IngestionConfig(methods=[DocumentIngestion(store=document)]),
        retrieval=RetrievalConfig(methods=[DocumentRetrieval(store=document)]),
    )
    engine = RagEngine(cfg)
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

    cfg = RagEngineConfig(
        metadata_store=metadata,
        ingestion=IngestionConfig(
            methods=[VectorIngestion(store=vector, embeddings=embeddings)],
        ),
        retrieval=RetrievalConfig(
            methods=[VectorRetrieval(store=vector, embeddings=embeddings)],
        ),
    )
    engine = RagEngine(cfg)
    await engine.initialize()

    await engine.shutdown()
    await engine.shutdown()

    assert vector.shutdown.await_count == 1
    assert metadata.shutdown.await_count == 1
