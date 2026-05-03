"""Multi-collection wiring tests.

Regression: `_retrieval_by_collection` and `_ingestion_by_collection` must be
populated symmetrically at initialize() time for every configured collection.
Passing an unknown collection name must raise, not silently fall back to the
default pipeline (which previously mixed data across collections)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rfnry_knowledge.config import IngestionConfig, KnowledgeEngineConfig, RetrievalConfig
from rfnry_knowledge.ingestion.methods.document import DocumentIngestion
from rfnry_knowledge.ingestion.methods.graph import GraphIngestion
from rfnry_knowledge.ingestion.methods.vector import VectorIngestion
from rfnry_knowledge.knowledge.engine import KnowledgeEngine
from rfnry_knowledge.retrieval.methods.document import DocumentRetrieval
from rfnry_knowledge.retrieval.methods.graph import GraphRetrieval
from rfnry_knowledge.retrieval.methods.vector import VectorRetrieval


def _make_vector_store(collections: list[str]) -> MagicMock:
    store = MagicMock()
    store.initialize = AsyncMock()
    store.shutdown = AsyncMock()
    store.collections = collections

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


def _vector_only_config(vector_store, embeddings) -> KnowledgeEngineConfig:
    return KnowledgeEngineConfig(
        ingestion=IngestionConfig(
            methods=[VectorIngestion(store=vector_store, embeddings=embeddings)],
        ),
        retrieval=RetrievalConfig(methods=[VectorRetrieval(store=vector_store, embeddings=embeddings)]),
    )


@pytest.mark.asyncio
async def test_initialize_populates_both_maps_for_every_collection():
    vector_store = _make_vector_store(["a", "b", "c"])
    embeddings = _make_embeddings()
    engine = KnowledgeEngine(_vector_only_config(vector_store, embeddings))
    await engine.initialize()

    assert set(engine._retrieval_by_collection.keys()) == {"a", "b", "c"}
    assert set(engine._ingestion_by_collection.keys()) == {"a", "b", "c"}


@pytest.mark.asyncio
async def test_get_ingestion_raises_on_unknown_collection():
    vector_store = _make_vector_store(["a"])
    embeddings = _make_embeddings()
    engine = KnowledgeEngine(_vector_only_config(vector_store, embeddings))
    await engine.initialize()

    with pytest.raises(ValueError, match="unknown collection"):
        engine._get_ingestion("nonexistent")


@pytest.mark.asyncio
async def test_get_retrieval_raises_on_unknown_collection():
    vector_store = _make_vector_store(["a"])
    embeddings = _make_embeddings()
    engine = KnowledgeEngine(_vector_only_config(vector_store, embeddings))
    await engine.initialize()

    with pytest.raises(ValueError, match="unknown collection"):
        engine._get_retrieval("nonexistent")


@pytest.mark.asyncio
async def test_non_default_collection_uses_scoped_store():
    vector_store = _make_vector_store(["a", "b"])
    embeddings = _make_embeddings()
    engine = KnowledgeEngine(_vector_only_config(vector_store, embeddings))
    await engine.initialize()

    vector_store.scoped.assert_called_with("b")


@pytest.mark.asyncio
async def test_default_collection_uses_unscoped_default_services():
    vector_store = _make_vector_store(["a"])
    embeddings = _make_embeddings()
    engine = KnowledgeEngine(_vector_only_config(vector_store, embeddings))
    await engine.initialize()

    assert engine._ingestion_by_collection["a"] is engine._ingestion_service


@pytest.mark.asyncio
async def test_cache_invalidation_fans_out_to_all_scoped_collections():
    """_on_source_removed and _on_ingestion_complete must invalidate BM25
    caches on every scoped collection, not just the default."""
    vector_store = _make_vector_store(["a", "b", "c"])
    embeddings = _make_embeddings()
    engine = KnowledgeEngine(_vector_only_config(vector_store, embeddings))
    await engine.initialize()

    invalidated: list[tuple[int, str | None]] = []

    for idx, (retrieval_service, _) in enumerate(engine._retrieval_by_collection.values()):
        for method in retrieval_service.methods:
            if method.name == "vector":

                async def _capture(knowledge_id, i=idx):
                    invalidated.append((i, knowledge_id))

                method.invalidate_cache = _capture  # type: ignore[method-assign]

    await engine._on_source_removed("kb-1")
    await engine._on_ingestion_complete("kb-2")

    assert len(invalidated) == 6
    seen_indexes = {i for i, _ in invalidated}
    assert seen_indexes == {0, 1, 2}


def _make_metadata_store() -> MagicMock:
    store = MagicMock()
    store.initialize = AsyncMock()
    store.shutdown = AsyncMock()
    store.list_sources = AsyncMock(return_value=[])
    return store


async def _build_multi_collection_engine() -> KnowledgeEngine:
    vector_store = _make_vector_store(["primary", "secondary"])
    embeddings = _make_embeddings()
    metadata_store = _make_metadata_store()

    from rfnry_knowledge.ingestion.methods.analyzed import AnalyzedIngestion

    engine = KnowledgeEngine(
        KnowledgeEngineConfig(
            metadata_store=metadata_store,
            ingestion=IngestionConfig(
                methods=[
                    VectorIngestion(store=vector_store, embeddings=embeddings),
                    AnalyzedIngestion(store=vector_store, embeddings=embeddings, vision=MagicMock()),
                ],
            ),
            retrieval=RetrievalConfig(methods=[VectorRetrieval(store=vector_store, embeddings=embeddings)]),
        )
    )
    await engine.initialize()
    return engine


async def test_ingest_structured_path_rejects_non_default_collection(tmp_path) -> None:
    engine = await _build_multi_collection_engine()
    xml_file = tmp_path / "sample.xml"
    xml_file.write_text("<root/>")
    with pytest.raises(ValueError, match="structured ingestion does not support collection routing"):
        await engine.ingest(xml_file, collection="secondary")
    await engine.shutdown()


async def test_analyze_rejects_non_default_collection(tmp_path) -> None:
    engine = await _build_multi_collection_engine()
    xml_file = tmp_path / "sample.xml"
    xml_file.write_text("<root/>")
    with pytest.raises(ValueError, match="structured ingestion does not support collection routing"):
        await engine.analyze(xml_file, collection="secondary")
    await engine.shutdown()


def _make_graph_store() -> MagicMock:
    store = MagicMock()
    store.initialize = AsyncMock()
    store.shutdown = AsyncMock()
    store.add_entities = AsyncMock()
    store.add_relations = AsyncMock()
    store.query_graph = AsyncMock(return_value=[])
    store.delete_by_source = AsyncMock()
    return store


async def _build_engine_with_all_methods(collections: list[str]) -> KnowledgeEngine:
    vector_store = _make_vector_store(collections)
    embeddings = _make_embeddings()
    metadata_store = _make_metadata_store()
    document_store = MagicMock()
    document_store.initialize = AsyncMock()
    document_store.shutdown = AsyncMock()
    graph_store = _make_graph_store()
    lm_client = MagicMock()

    _patches = [
        patch("rfnry_knowledge.ingestion.methods.graph.build_registry", return_value=MagicMock()),
        patch("rfnry_knowledge.ingestion.analyze.service.build_registry", return_value=MagicMock()),
        patch("rfnry_knowledge.knowledge.engine.build_registry", return_value=MagicMock()),
    ]
    for p in _patches:
        p.start()

    engine = KnowledgeEngine(
        KnowledgeEngineConfig(
            metadata_store=metadata_store,
            ingestion=IngestionConfig(
                methods=[
                    VectorIngestion(store=vector_store, embeddings=embeddings),
                    DocumentIngestion(store=document_store),
                    GraphIngestion(store=graph_store, provider_client=lm_client),
                ],
            ),
            retrieval=RetrievalConfig(
                methods=[
                    VectorRetrieval(store=vector_store, embeddings=embeddings),
                    DocumentRetrieval(store=document_store),
                    GraphRetrieval(store=graph_store),
                ],
            ),
        )
    )
    try:
        await engine.initialize()
    finally:
        for p in _patches:
            p.stop()
    return engine


async def test_scoped_ingestion_pipeline_includes_graph(tmp_path) -> None:
    """Non-default collection must get GraphIngestion when configured."""
    engine = await _build_engine_with_all_methods(collections=["primary", "secondary"])
    secondary_svc = engine._ingestion_by_collection["secondary"]
    method_types = {type(m).__name__ for m in secondary_svc._ingestion_methods}
    assert "VectorIngestion" in method_types
    assert "DocumentIngestion" in method_types
    assert "GraphIngestion" in method_types
    await engine.shutdown()
