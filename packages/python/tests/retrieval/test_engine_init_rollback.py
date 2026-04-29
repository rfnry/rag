"""Regression: RagEngine.initialize() must roll back opened stores on partial failure.

Before the fix, if graph_store.initialize() raised after metadata_store and
document_store had already opened connections, those connections leaked because
__aexit__ doesn't fire when __aenter__ raises, and shutdown() was not called."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from rfnry_rag.retrieval.server import (
    IngestionConfig,
    PersistenceConfig,
    RagEngine,
    RagServerConfig,
)


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

    cfg = RagServerConfig(
        persistence=PersistenceConfig(
            metadata_store=metadata_store,
            document_store=document_store,
            graph_store=graph_store,
        ),
        ingestion=IngestionConfig(),
    )
    engine = RagEngine(cfg)

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

    cfg = RagServerConfig(
        persistence=PersistenceConfig(document_store=document_store),
        ingestion=IngestionConfig(),
    )
    engine = RagEngine(cfg)
    await engine.initialize()

    document_store.shutdown.assert_not_called()
    assert engine._initialized is True
