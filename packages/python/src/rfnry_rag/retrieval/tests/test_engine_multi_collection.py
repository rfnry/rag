"""Multi-collection wiring tests.

Regression: `_retrieval_by_collection` and `_ingestion_by_collection` must be
populated symmetrically at initialize() time for every configured collection.
Passing an unknown collection name must raise, not silently fall back to the
default pipeline (which previously mixed data across collections)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from rfnry_rag.retrieval.server import (
    IngestionConfig,
    PersistenceConfig,
    RagEngine,
    RagServerConfig,
)


def _make_vector_store(collections: list[str]) -> MagicMock:
    """MagicMock that quacks like a multi-collection QdrantVectorStore."""
    store = MagicMock()
    store.initialize = AsyncMock()
    store.shutdown = AsyncMock()
    store.collections = collections

    # scoped(name) returns a separate mock so we can tell them apart.
    def _scoped(name: str) -> MagicMock:
        scoped = MagicMock()
        scoped.initialize = AsyncMock()
        scoped.shutdown = AsyncMock()
        scoped._pinned = name
        return scoped

    store.scoped = MagicMock(side_effect=_scoped)
    return store


def _make_embeddings() -> MagicMock:
    embeddings = MagicMock()
    embeddings.model = "test"
    embeddings.embedding_dimension = AsyncMock(return_value=1536)
    return embeddings


@pytest.mark.asyncio
async def test_initialize_populates_both_maps_for_every_collection():
    vector_store = _make_vector_store(["a", "b", "c"])
    embeddings = _make_embeddings()

    engine = RagEngine(
        RagServerConfig(
            persistence=PersistenceConfig(vector_store=vector_store),
            ingestion=IngestionConfig(embeddings=embeddings),
        )
    )
    await engine.initialize()

    assert set(engine._retrieval_by_collection.keys()) == {"a", "b", "c"}
    assert set(engine._ingestion_by_collection.keys()) == {"a", "b", "c"}


@pytest.mark.asyncio
async def test_get_ingestion_raises_on_unknown_collection():
    vector_store = _make_vector_store(["a"])
    embeddings = _make_embeddings()
    engine = RagEngine(
        RagServerConfig(
            persistence=PersistenceConfig(vector_store=vector_store),
            ingestion=IngestionConfig(embeddings=embeddings),
        )
    )
    await engine.initialize()

    with pytest.raises(ValueError, match="unknown collection"):
        engine._get_ingestion("nonexistent")


@pytest.mark.asyncio
async def test_get_retrieval_raises_on_unknown_collection():
    vector_store = _make_vector_store(["a"])
    embeddings = _make_embeddings()
    engine = RagEngine(
        RagServerConfig(
            persistence=PersistenceConfig(vector_store=vector_store),
            ingestion=IngestionConfig(embeddings=embeddings),
        )
    )
    await engine.initialize()

    with pytest.raises(ValueError, match="unknown collection"):
        engine._get_retrieval("nonexistent")


@pytest.mark.asyncio
async def test_non_default_collection_uses_scoped_store():
    vector_store = _make_vector_store(["a", "b"])
    embeddings = _make_embeddings()
    engine = RagEngine(
        RagServerConfig(
            persistence=PersistenceConfig(vector_store=vector_store),
            ingestion=IngestionConfig(embeddings=embeddings),
        )
    )
    await engine.initialize()

    # scoped() must have been called once for "b" (not for "a", which reuses defaults)
    vector_store.scoped.assert_called_with("b")


@pytest.mark.asyncio
async def test_default_collection_uses_unscoped_default_services():
    vector_store = _make_vector_store(["a"])
    embeddings = _make_embeddings()
    engine = RagEngine(
        RagServerConfig(
            persistence=PersistenceConfig(vector_store=vector_store),
            ingestion=IngestionConfig(embeddings=embeddings),
        )
    )
    await engine.initialize()

    # The first collection should reuse the default service instances, not scoped ones.
    assert engine._ingestion_by_collection["a"] is engine._ingestion_service


@pytest.mark.asyncio
async def test_cache_invalidation_fans_out_to_all_scoped_collections():
    """_on_source_removed and _on_ingestion_complete must invalidate BM25
    caches on every scoped collection, not just the default."""
    vector_store = _make_vector_store(["a", "b", "c"])
    embeddings = _make_embeddings()
    engine = RagEngine(
        RagServerConfig(
            persistence=PersistenceConfig(vector_store=vector_store),
            ingestion=IngestionConfig(embeddings=embeddings),
        )
    )
    await engine.initialize()

    # Replace each collection's VectorRetrieval with a spy that records invalidations.
    invalidated: list[tuple[int, str | None]] = []

    for idx, (retrieval_service, _) in enumerate(engine._retrieval_by_collection.values()):
        for method in retrieval_service._retrieval_methods:
            if method.name == "vector":

                async def _capture(knowledge_id, i=idx):
                    invalidated.append((i, knowledge_id))

                method.invalidate_cache = _capture  # type: ignore[method-assign]

    await engine._on_source_removed("kb-1")
    await engine._on_ingestion_complete("kb-2")

    # Three collections × two callbacks = 6 invalidations
    assert len(invalidated) == 6
    seen_indexes = {i for i, _ in invalidated}
    assert seen_indexes == {0, 1, 2}
