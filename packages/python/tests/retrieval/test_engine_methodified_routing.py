"""RagEngine routes drawing/analyzed via cfg.ingestion.methods isinstance lookup."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from rfnry_rag.config.drawing import DrawingIngestionConfig
from rfnry_rag.config.engine import RagEngineConfig
from rfnry_rag.config.ingestion import IngestionConfig
from rfnry_rag.config.retrieval import RetrievalConfig
from rfnry_rag.ingestion.methods.analyzed import AnalyzedIngestion
from rfnry_rag.ingestion.methods.drawing import DrawingIngestion
from rfnry_rag.ingestion.methods.vector import VectorIngestion
from rfnry_rag.retrieval.methods.vector import VectorRetrieval
from rfnry_rag.server import RagEngine


def _make_metadata_store() -> MagicMock:
    metadata_store = MagicMock()
    metadata_store.initialize = AsyncMock()
    metadata_store.shutdown = AsyncMock()
    metadata_store.list_sources = AsyncMock(return_value=[])
    metadata_store.find_by_hash = AsyncMock(return_value=None)
    return metadata_store


def _make_vector_store(collections: list[str] | None = None) -> MagicMock:
    store = MagicMock()
    store.initialize = AsyncMock()
    store.shutdown = AsyncMock()
    store.collections = collections if collections is not None else []
    return store


def _make_embeddings(dim: int = 1536) -> MagicMock:
    embeddings = MagicMock()
    embeddings.model = "test-emb"
    embeddings.name = "test:emb"
    embeddings.embedding_dimension = AsyncMock(return_value=dim)
    return embeddings


def _engine_config(
    *,
    metadata_store: Any,
    vector_store: Any,
    embeddings: Any,
    ingestion_methods: list[Any],
) -> RagEngineConfig:
    return RagEngineConfig(
        metadata_store=metadata_store,
        ingestion=IngestionConfig(methods=ingestion_methods),
        retrieval=RetrievalConfig(
            methods=[VectorRetrieval(store=vector_store, embeddings=embeddings)],
        ),
    )


@pytest.mark.asyncio
async def test_engine_picks_up_analyzed_method_from_methods_list() -> None:
    """When cfg.ingestion.methods carries AnalyzedIngestion, the engine wires
    its inner service as ``self._structured_ingestion``."""
    metadata_store = _make_metadata_store()
    vector_store = _make_vector_store()
    embeddings = _make_embeddings()

    analyzed = AnalyzedIngestion(
        store=vector_store,
        embeddings=embeddings,
        vision=MagicMock(),
    )
    cfg = _engine_config(
        metadata_store=metadata_store,
        vector_store=vector_store,
        embeddings=embeddings,
        ingestion_methods=[
            VectorIngestion(store=vector_store, embeddings=embeddings),
            analyzed,
        ],
    )
    engine = RagEngine(cfg)
    await engine.initialize()
    try:
        assert engine._structured_ingestion is analyzed._service_ref()
        assert engine._analyzed_method is analyzed
    finally:
        await engine.shutdown()


@pytest.mark.asyncio
async def test_engine_picks_up_drawing_method_from_methods_list() -> None:
    """When cfg.ingestion.methods carries DrawingIngestion, the engine wires
    its inner service as ``self._drawing_ingestion``."""
    metadata_store = _make_metadata_store()
    vector_store = _make_vector_store()
    embeddings = _make_embeddings()

    drawing = DrawingIngestion(
        config=DrawingIngestionConfig(enabled=True),
        store=vector_store,
        embeddings=embeddings,
        vision=MagicMock(),
    )
    cfg = _engine_config(
        metadata_store=metadata_store,
        vector_store=vector_store,
        embeddings=embeddings,
        ingestion_methods=[
            VectorIngestion(store=vector_store, embeddings=embeddings),
            drawing,
        ],
    )
    engine = RagEngine(cfg)
    await engine.initialize()
    try:
        assert engine._drawing_ingestion is drawing._service_ref()
        assert engine._drawing_method is drawing
    finally:
        await engine.shutdown()


@pytest.mark.asyncio
async def test_engine_routing_uses_method_accepts_when_method_present() -> None:
    """When a DrawingIngestion method is configured, the ingest() routing
    consults ``method.accepts()`` instead of the legacy extension-set check.

    .pdf with source_type='drawing' routes to drawing; .pdf with source_type=None
    does not (it falls through to standard or analyzed)."""
    drawing_service = SimpleNamespace(
        render=AsyncMock(side_effect=_async_source_factory(status="rendered")),
        extract=AsyncMock(side_effect=_async_source_factory(status="extracted")),
        link=AsyncMock(side_effect=_async_source_factory(status="linked")),
        ingest=AsyncMock(side_effect=_async_source_factory(status="completed")),
    )
    drawing_method = MagicMock()
    drawing_method.accepts = MagicMock(side_effect=lambda fp, st: st == "drawing")

    fast_ingest = AsyncMock(side_effect=_async_source_factory(status="completed"))

    engine = RagEngine.__new__(RagEngine)
    engine._initialized = True  # type: ignore[attr-defined]
    engine._drawing_ingestion = drawing_service  # type: ignore[attr-defined]
    engine._drawing_method = drawing_method  # type: ignore[attr-defined]
    engine._structured_ingestion = None  # type: ignore[attr-defined]
    engine._analyzed_method = None  # type: ignore[attr-defined]
    engine._config = SimpleNamespace(metadata_store=None)  # type: ignore[assignment]
    engine._get_ingestion = lambda collection: SimpleNamespace(ingest=fast_ingest)  # type: ignore[assignment,method-assign]

    await engine.ingest("/tmp/doc.pdf", knowledge_id="k", source_type="drawing")
    assert drawing_service.render.await_count == 1
    drawing_method.accepts.assert_called()
    assert fast_ingest.await_count == 0

    await engine.ingest("/tmp/doc.pdf", knowledge_id="k")
    assert drawing_service.render.await_count == 1
    assert fast_ingest.await_count == 1


def _async_source_factory(*, status: str, source_id: str = "src-1", file_hash: str = "h"):
    from rfnry_rag.models import Source

    async def _return(*args: Any, **kwargs: Any) -> Source:
        sid = args[0] if args and isinstance(args[0], str) and not args[0].startswith("/") else source_id
        return Source(
            source_id=sid,
            status=status,
            file_hash=file_hash,
            knowledge_id=kwargs.get("knowledge_id") or "k",
            metadata={"source_format": "pdf"},
        )

    return _return
