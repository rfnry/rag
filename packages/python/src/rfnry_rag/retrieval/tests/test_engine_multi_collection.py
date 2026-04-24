"""Multi-collection wiring tests.

Regression: `_retrieval_by_collection` and `_ingestion_by_collection` must be
populated symmetrically at initialize() time for every configured collection.
Passing an unknown collection name must raise, not silently fall back to the
default pipeline (which previously mixed data across collections)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rfnry_rag.retrieval.server import (
    IngestionConfig,
    PersistenceConfig,
    RagEngine,
    RagServerConfig,
    TreeIndexingConfig,
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
        for method in retrieval_service.methods:
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


# ---------------------------------------------------------------------------
# Helpers for structured-ingestion collection-routing tests (C1)
# ---------------------------------------------------------------------------


def _make_metadata_store() -> MagicMock:
    store = MagicMock()
    store.initialize = AsyncMock()
    store.shutdown = AsyncMock()
    store.list_sources = AsyncMock(return_value=[])
    return store


async def _build_multi_collection_engine() -> RagEngine:
    """Build a minimal RagEngine that has _structured_ingestion wired up.

    Requires: metadata_store + vector_store + embeddings (see server.py:567).
    Uses two collections so collection= routing is meaningful.
    """
    vector_store = _make_vector_store(["primary", "secondary"])
    embeddings = _make_embeddings()
    metadata_store = _make_metadata_store()

    engine = RagEngine(
        RagServerConfig(
            persistence=PersistenceConfig(
                vector_store=vector_store,
                metadata_store=metadata_store,
            ),
            ingestion=IngestionConfig(embeddings=embeddings),
        )
    )
    await engine.initialize()
    return engine


async def test_ingest_structured_path_rejects_non_default_collection(tmp_path) -> None:
    """ingest() with .xml/.l5x + collection= must raise, not silently write to default."""
    rag = await _build_multi_collection_engine()
    xml_file = tmp_path / "sample.xml"
    xml_file.write_text("<root/>")
    with pytest.raises(ValueError, match="structured ingestion does not support collection routing"):
        await rag.ingest(xml_file, collection="secondary")
    await rag.shutdown()


async def test_analyze_rejects_non_default_collection(tmp_path) -> None:
    rag = await _build_multi_collection_engine()
    xml_file = tmp_path / "sample.xml"
    xml_file.write_text("<root/>")
    with pytest.raises(ValueError, match="structured ingestion does not support collection routing"):
        await rag.analyze(xml_file, collection="secondary")
    await rag.shutdown()


# ---------------------------------------------------------------------------
# M1 regression: scoped-collection ingestion must mirror full method assembly
# ---------------------------------------------------------------------------


def _make_graph_store() -> MagicMock:
    store = MagicMock()
    store.initialize = AsyncMock()
    store.shutdown = AsyncMock()
    store.add_entities = AsyncMock()
    store.add_relations = AsyncMock()
    store.query_graph = AsyncMock(return_value=[])
    store.delete_by_source = AsyncMock()
    return store


async def _build_engine_with_all_methods(collections: list[str]) -> RagEngine:
    """Build an engine with graph store and tree indexing enabled.

    All four ingestion methods (vector, document, graph, tree) should be
    assembled for every collection — including non-default ones.

    build_registry is patched in GraphIngestion to avoid touching the BAML
    C extension, which rejects MagicMock provider objects.
    """
    vector_store = _make_vector_store(collections)
    embeddings = _make_embeddings()
    metadata_store = _make_metadata_store()
    document_store = MagicMock()
    document_store.initialize = AsyncMock()
    document_store.shutdown = AsyncMock()
    graph_store = _make_graph_store()
    lm_client = MagicMock()  # GraphIngestion only calls it at ingest time, not at init

    engine = RagEngine(
        RagServerConfig(
            persistence=PersistenceConfig(
                vector_store=vector_store,
                metadata_store=metadata_store,
                document_store=document_store,
                graph_store=graph_store,
            ),
            ingestion=IngestionConfig(
                embeddings=embeddings,
                lm_client=lm_client,
            ),
            tree_indexing=TreeIndexingConfig(enabled=True, model=MagicMock()),
        )
    )
    # Patch build_registry in every module that calls it at construction time
    # during initialize() — GraphIngestion, AnalyzedIngestionService, and the
    # tree-indexing/tree-search paths in server.py all call it in __init__ or
    # during _initialize_impl, which rejects MagicMock provider objects via
    # BAML's C extension.
    _patches = [
        patch("rfnry_rag.retrieval.modules.ingestion.methods.graph.build_registry", return_value=MagicMock()),
        patch("rfnry_rag.retrieval.modules.ingestion.analyze.service.build_registry", return_value=MagicMock()),
        patch("rfnry_rag.retrieval.server.build_registry", return_value=MagicMock()),
    ]
    for p in _patches:
        p.start()
    try:
        await engine.initialize()
    finally:
        for p in _patches:
            p.stop()
    return engine


async def test_scoped_ingestion_pipeline_includes_graph_and_tree(tmp_path) -> None:
    """Non-default collection must get GraphIngestion + TreeIngestion when configured."""
    rag = await _build_engine_with_all_methods(collections=["primary", "secondary"])
    secondary_svc = rag._ingestion_by_collection["secondary"]
    method_types = {type(m).__name__ for m in secondary_svc._ingestion_methods}
    assert "VectorIngestion" in method_types
    assert "DocumentIngestion" in method_types
    assert "GraphIngestion" in method_types
    assert "TreeIngestion" in method_types
    await rag.shutdown()
