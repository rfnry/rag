"""RagEngine drawing routing: .dxf always; .pdf only with source_type='drawing'."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from rfnry_rag.config.drawing import DrawingIngestionConfig
from rfnry_rag.config.engine import RagEngineConfig
from rfnry_rag.config.ingestion import IngestionConfig
from rfnry_rag.config.retrieval import RetrievalConfig
from rfnry_rag.exceptions import ConfigurationError
from rfnry_rag.ingestion.methods.drawing import DrawingIngestion
from rfnry_rag.ingestion.methods.vector import VectorIngestion
from rfnry_rag.retrieval.methods.vector import VectorRetrieval
from rfnry_rag.server import SUPPORTED_DRAWING_EXTENSIONS, RagEngine


def test_drawing_extensions_allowlist_is_dxf_only() -> None:
    """Only .dxf auto-routes to drawing; .pdf is tiebroken via source_type='drawing'."""
    assert ".dxf" in SUPPORTED_DRAWING_EXTENSIONS
    assert ".pdf" not in SUPPORTED_DRAWING_EXTENSIONS
    from rfnry_rag.server import SUPPORTED_STRUCTURED_EXTENSIONS

    assert {".xml", ".l5x"} == SUPPORTED_STRUCTURED_EXTENSIONS


def test_engine_exposes_stepped_drawing_methods() -> None:
    """Stepped API mirrors analyzed service: render/extract/link/complete_ingestion."""
    assert hasattr(RagEngine, "render_drawing")
    assert hasattr(RagEngine, "extract_drawing")
    assert hasattr(RagEngine, "link_drawing")
    assert hasattr(RagEngine, "complete_drawing_ingestion")


class _StubDrawingService:
    """Stub for DrawingIngestionService — records phase calls."""

    def __init__(self) -> None:
        self.render = AsyncMock()
        self.extract = AsyncMock()
        self.link = AsyncMock()
        self.ingest = AsyncMock()


def _make_metadata_store() -> MagicMock:
    metadata_store = MagicMock()
    metadata_store.initialize = AsyncMock()
    metadata_store.shutdown = AsyncMock()
    metadata_store.list_sources = AsyncMock(return_value=[])
    metadata_store.find_by_hash = AsyncMock(return_value=None)
    return metadata_store


def _make_vector_store() -> MagicMock:
    store = MagicMock()
    store.initialize = AsyncMock()
    store.shutdown = AsyncMock()
    store.collections = []
    return store


def _make_embeddings(dim: int = 4) -> MagicMock:
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


async def _make_engine_with_drawing_stub(stub: _StubDrawingService) -> RagEngine:
    """Build a real RagEngine wired with a DrawingIngestion method, then swap
    the engine's cached drawing service for a phase-call stub."""
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
    engine._drawing_ingestion = stub  # type: ignore[assignment]
    return engine


async def _make_engine_without_drawing() -> RagEngine:
    metadata_store = _make_metadata_store()
    vector_store = _make_vector_store()
    embeddings = _make_embeddings()
    cfg = _engine_config(
        metadata_store=metadata_store,
        vector_store=vector_store,
        embeddings=embeddings,
        ingestion_methods=[VectorIngestion(store=vector_store, embeddings=embeddings)],
    )
    engine = RagEngine(cfg)
    await engine.initialize()
    return engine


async def test_stepped_methods_raise_when_drawing_service_not_configured() -> None:
    """Each stepped method raises ConfigurationError with a useful message when
    DrawingIngestionConfig is not configured."""
    engine = await _make_engine_without_drawing()
    try:
        with pytest.raises(ConfigurationError, match="(?i)drawing"):
            await engine.render_drawing("/tmp/x.dxf", knowledge_id="k")
        with pytest.raises(ConfigurationError, match="(?i)drawing"):
            await engine.extract_drawing("src")
        with pytest.raises(ConfigurationError, match="(?i)drawing"):
            await engine.link_drawing("src")
        with pytest.raises(ConfigurationError, match="(?i)drawing"):
            await engine.complete_drawing_ingestion("src")
    finally:
        await engine.shutdown()


async def test_ingest_routes_dxf_to_drawing_service(tmp_path: Any) -> None:
    """A .dxf file dispatches to _drawing_ingestion, not the fast path."""
    dxf_path = tmp_path / "fake.dxf"
    dxf_path.write_bytes(b"fake-dxf")

    stub = _StubDrawingService()
    stub.render.side_effect = _async_source_factory(status="rendered")
    stub.extract.side_effect = _async_source_factory(status="extracted")
    stub.link.side_effect = _async_source_factory(status="linked")
    stub.ingest.side_effect = _async_source_factory(status="completed")

    engine = await _make_engine_with_drawing_stub(stub)
    try:
        await engine.ingest(str(dxf_path), knowledge_id="k")
    finally:
        await engine.shutdown()

    assert stub.render.await_count == 1
    assert stub.extract.await_count == 1
    assert stub.link.await_count == 1
    assert stub.ingest.await_count == 1


async def test_ingest_routes_pdf_to_drawing_only_with_source_type(tmp_path: Any) -> None:
    """A .pdf file without source_type='drawing' stays on the fast path."""
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"fake-pdf")

    stub = _StubDrawingService()
    engine = await _make_engine_with_drawing_stub(stub)

    fast_ingest = AsyncMock(side_effect=_async_source_factory(status="completed"))
    engine._get_ingestion = lambda collection: SimpleNamespace(ingest=fast_ingest)  # type: ignore[assignment,method-assign]

    try:
        await engine.ingest(str(pdf_path), knowledge_id="k")
        assert stub.render.await_count == 0
        assert fast_ingest.await_count == 1

        stub.render.side_effect = _async_source_factory(status="rendered")
        stub.extract.side_effect = _async_source_factory(status="extracted")
        stub.link.side_effect = _async_source_factory(status="linked")
        stub.ingest.side_effect = _async_source_factory(status="completed")
        await engine.ingest(str(pdf_path), knowledge_id="k", source_type="drawing")
        assert stub.render.await_count == 1
    finally:
        await engine.shutdown()


async def test_ingest_dxf_raises_when_drawing_service_not_configured(tmp_path: Any) -> None:
    """A .dxf file without a configured drawing service is a user error."""
    dxf_path = tmp_path / "x.dxf"
    dxf_path.write_bytes(b"fake-dxf")

    engine = await _make_engine_without_drawing()
    try:
        with pytest.raises(ValueError, match="(?i)drawing"):
            await engine.ingest(str(dxf_path), knowledge_id="k")
    finally:
        await engine.shutdown()


async def test_ingest_drawing_rejects_collection_argument(tmp_path: Any) -> None:
    dxf_path = tmp_path / "x.dxf"
    dxf_path.write_bytes(b"fake-dxf")

    stub = _StubDrawingService()
    engine = await _make_engine_with_drawing_stub(stub)
    try:
        with pytest.raises(ValueError, match="(?i)collection"):
            await engine.ingest(str(dxf_path), knowledge_id="k", collection="other")
    finally:
        await engine.shutdown()


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
