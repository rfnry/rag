"""RagEngine drawing routing: .dxf always; .pdf only with source_type='drawing'."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from rfnry_rag.common.errors import ConfigurationError
from rfnry_rag.server import SUPPORTED_DRAWING_EXTENSIONS


def test_drawing_extensions_allowlist_is_dxf_only() -> None:
    """Only .dxf auto-routes to drawing; .pdf is tiebroken via source_type='drawing'."""
    assert ".dxf" in SUPPORTED_DRAWING_EXTENSIONS
    assert ".pdf" not in SUPPORTED_DRAWING_EXTENSIONS
    # Structured allowlist should also remain .xml + .l5x only (no .pdf regression)
    from rfnry_rag.server import SUPPORTED_STRUCTURED_EXTENSIONS

    assert {".xml", ".l5x"} == SUPPORTED_STRUCTURED_EXTENSIONS


class _StubIngestionService:
    """Stub for DrawingIngestionService — records phase calls."""

    def __init__(self) -> None:
        self.render = AsyncMock()
        self.extract = AsyncMock()
        self.link = AsyncMock()
        self.ingest = AsyncMock()


def test_engine_exposes_stepped_drawing_methods() -> None:
    """Stepped API mirrors analyzed service: render/extract/link/complete_ingestion."""
    from rfnry_rag.server import RagEngine

    assert hasattr(RagEngine, "render_drawing")
    assert hasattr(RagEngine, "extract_drawing")
    assert hasattr(RagEngine, "link_drawing")
    assert hasattr(RagEngine, "complete_drawing_ingestion")


async def test_stepped_methods_raise_when_drawing_service_not_configured() -> None:
    """Each stepped method raises ConfigurationError with a useful message when
    DrawingIngestionConfig is not configured."""
    from rfnry_rag.server import RagEngine

    engine = RagEngine.__new__(RagEngine)
    engine._drawing_ingestion = None  # type: ignore[attr-defined]
    engine._initialized = True  # type: ignore[attr-defined]

    with pytest.raises(ConfigurationError, match="(?i)drawing"):
        await engine.render_drawing("/tmp/x.dxf", knowledge_id="k")
    with pytest.raises(ConfigurationError, match="(?i)drawing"):
        await engine.extract_drawing("src")
    with pytest.raises(ConfigurationError, match="(?i)drawing"):
        await engine.link_drawing("src")
    with pytest.raises(ConfigurationError, match="(?i)drawing"):
        await engine.complete_drawing_ingestion("src")


async def test_ingest_routes_dxf_to_drawing_service() -> None:
    """A .dxf file dispatches to _drawing_ingestion, not the fast path."""
    stub = _StubIngestionService()
    stub.render.side_effect = _async_source_factory(status="rendered")
    stub.extract.side_effect = _async_source_factory(status="extracted")
    stub.link.side_effect = _async_source_factory(status="linked")
    stub.ingest.side_effect = _async_source_factory(status="completed")

    engine = _make_engine_stub_with_drawing(stub)
    await engine.ingest("/tmp/fake.dxf", knowledge_id="k")

    assert stub.render.await_count == 1
    assert stub.extract.await_count == 1
    assert stub.link.await_count == 1
    assert stub.ingest.await_count == 1


async def test_ingest_routes_pdf_to_drawing_only_with_source_type() -> None:
    """A .pdf file without source_type='drawing' stays on the fast path."""
    stub = _StubIngestionService()
    engine = _make_engine_stub_with_drawing(stub)

    # Fast-path: fallback dispatcher. We short-circuit it with a mock too.
    fast_ingest = AsyncMock(side_effect=_async_source_factory(status="completed"))
    engine._get_ingestion = lambda collection: SimpleNamespace(ingest=fast_ingest)  # type: ignore[assignment,method-assign,return-value]

    # Without source_type -> fast path
    await engine.ingest("/tmp/doc.pdf", knowledge_id="k")
    assert stub.render.await_count == 0
    assert fast_ingest.await_count == 1

    # With source_type='drawing' -> drawing path
    stub.render.side_effect = _async_source_factory(status="rendered")
    stub.extract.side_effect = _async_source_factory(status="extracted")
    stub.link.side_effect = _async_source_factory(status="linked")
    stub.ingest.side_effect = _async_source_factory(status="completed")
    await engine.ingest("/tmp/doc.pdf", knowledge_id="k", source_type="drawing")
    assert stub.render.await_count == 1


async def test_ingest_dxf_raises_when_drawing_service_not_configured() -> None:
    """A .dxf file without a configured drawing service is a user error."""
    from rfnry_rag.server import RagEngine

    engine = RagEngine.__new__(RagEngine)
    engine._drawing_ingestion = None  # type: ignore[attr-defined]
    engine._initialized = True  # type: ignore[attr-defined]
    engine._structured_ingestion = None  # type: ignore[attr-defined]
    engine._config = SimpleNamespace(  # type: ignore[assignment]
        persistence=SimpleNamespace(metadata_store=None),
    )

    with pytest.raises(ValueError, match="(?i)drawing"):
        await engine.ingest("/tmp/x.dxf", knowledge_id="k")


async def test_ingest_drawing_rejects_collection_argument() -> None:
    stub = _StubIngestionService()
    engine = _make_engine_stub_with_drawing(stub)
    with pytest.raises(ValueError, match="(?i)collection"):
        await engine.ingest("/tmp/x.dxf", knowledge_id="k", collection="other")


# ---- helpers ----


def _async_source_factory(*, status: str, source_id: str = "src-1", file_hash: str = "h"):
    from rfnry_rag.retrieval.common.models import Source

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


def _make_engine_stub_with_drawing(drawing_service: Any):
    """Bypass RagEngine.initialize() and wire just enough state for routing tests."""
    from rfnry_rag.server import RagEngine

    engine = RagEngine.__new__(RagEngine)
    engine._drawing_ingestion = drawing_service  # type: ignore[attr-defined]
    engine._structured_ingestion = None  # type: ignore[attr-defined]
    engine._initialized = True  # type: ignore[attr-defined]
    # Avoid the fast-path by routing via a lambda
    engine._get_ingestion = lambda collection: SimpleNamespace(  # type: ignore[assignment,method-assign,return-value]
        ingest=AsyncMock(side_effect=_async_source_factory(status="completed")),
    )
    engine._config = SimpleNamespace(  # type: ignore[assignment]
        persistence=SimpleNamespace(metadata_store=None),
    )
    return engine
