"""Regression: KnowledgeEngine.initialize() must roll back opened stores on partial failure.

Before the fix, if graph_store.initialize() raised after metadata_store and
document_store had already opened connections, those connections leaked because
__aexit__ doesn't fire when __aenter__ raises, and shutdown() was not called."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from rfnry_knowledge.config import IngestionConfig, KnowledgeEngineConfig, RetrievalConfig
from rfnry_knowledge.ingestion.methods.document import DocumentIngestion
from rfnry_knowledge.ingestion.methods.graph import GraphIngestion
from rfnry_knowledge.knowledge.engine import KnowledgeEngine
from rfnry_knowledge.retrieval.methods.document import DocumentRetrieval
from rfnry_knowledge.retrieval.methods.graph import GraphRetrieval


@pytest.mark.asyncio
async def test_initialize_rolls_back_already_opened_stores_on_failure():
    metadata_store = MagicMock()
    metadata_store.initialize = AsyncMock()
    metadata_store.shutdown = AsyncMock()
    metadata_store.list_sources = AsyncMock(return_value=[])

    document_store = MagicMock()
    document_store.initialize = AsyncMock()
    document_store.shutdown = AsyncMock()

    graph_store = MagicMock()
    graph_store.initialize = AsyncMock(side_effect=RuntimeError("graph failed"))
    graph_store.shutdown = AsyncMock()

    cfg = KnowledgeEngineConfig(
        metadata_store=metadata_store,
        ingestion=IngestionConfig(
            methods=[DocumentIngestion(store=document_store), GraphIngestion(store=graph_store)],
        ),
        retrieval=RetrievalConfig(
            methods=[DocumentRetrieval(store=document_store), GraphRetrieval(store=graph_store)],
        ),
    )
    engine = KnowledgeEngine(cfg)

    with pytest.raises(RuntimeError, match="graph failed"):
        await engine.initialize()

    metadata_store.shutdown.assert_awaited()
    document_store.shutdown.assert_awaited()
    assert engine._initialized is False


@pytest.mark.asyncio
async def test_initialize_does_not_call_shutdown_on_success():
    """Sanity: successful init does not invoke shutdown()."""
    document_store = MagicMock()
    document_store.initialize = AsyncMock()
    document_store.shutdown = AsyncMock()

    cfg = KnowledgeEngineConfig(
        ingestion=IngestionConfig(methods=[DocumentIngestion(store=document_store)]),
        retrieval=RetrievalConfig(methods=[DocumentRetrieval(store=document_store)]),
    )
    engine = KnowledgeEngine(cfg)
    await engine.initialize()

    document_store.shutdown.assert_not_called()
    assert engine._initialized is True
