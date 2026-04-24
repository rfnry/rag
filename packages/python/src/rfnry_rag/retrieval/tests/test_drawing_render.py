"""Drawing render phase: PDF via PyMuPDF, DXF via ezdxf."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from rfnry_rag.retrieval.common.errors import IngestionError
from rfnry_rag.retrieval.modules.ingestion.drawing.config import DrawingIngestionConfig
from rfnry_rag.retrieval.modules.ingestion.drawing.service import DrawingIngestionService


class _InMemoryMetadataStore:
    """Minimal metadata store stand-in for render-phase tests."""

    def __init__(self) -> None:
        self._sources: dict[str, Any] = {}
        self._pages: dict[str, list[dict]] = {}

    async def create_source(self, source) -> None:
        self._sources[source.source_id] = source

    async def update_source(self, source_id: str, **fields) -> None:
        src = self._sources[source_id]
        for k, v in fields.items():
            setattr(src, k, v)

    async def find_by_hash(self, hash_value: str, knowledge_id: str | None):
        for s in self._sources.values():
            if s.file_hash == hash_value and s.knowledge_id == knowledge_id:
                return s
        return None

    async def upsert_page_analyses(self, source_id: str, analyses: list[dict]) -> None:
        self._pages[source_id] = list(analyses)

    async def get_page_analyses(self, source_id: str) -> list[dict]:
        return list(self._pages.get(source_id, []))


def _make_service(
    config: DrawingIngestionConfig | None = None,
) -> tuple[DrawingIngestionService, _InMemoryMetadataStore]:
    metadata = _InMemoryMetadataStore()
    cfg = config or DrawingIngestionConfig(enabled=True)
    svc = DrawingIngestionService(
        config=cfg,
        embeddings=SimpleNamespace(),  # type: ignore[arg-type]  # not used in render
        vector_store=SimpleNamespace(),  # type: ignore[arg-type]
        metadata_store=metadata,  # type: ignore[arg-type]
        embedding_model_name="test-embed",
    )
    return svc, metadata


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    """Create a 2-page PDF via PyMuPDF (no external fixture file)."""
    import pymupdf

    path = tmp_path / "sample.pdf"
    doc = pymupdf.open()
    for i in range(2):
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 72), f"Page {i + 1}")
    doc.save(path)
    doc.close()
    return path


@pytest.fixture
def sample_dxf(tmp_path: Path) -> Path:
    """Create a minimal DXF with a couple of INSERT entities via ezdxf."""
    import ezdxf

    path = tmp_path / "sample.dxf"
    doc = ezdxf.new()
    msp = doc.modelspace()
    msp.add_line((0, 0), (10, 0))
    msp.add_line((10, 0), (10, 10))
    doc.saveas(path)
    return path


async def test_render_pdf_produces_per_page_images(sample_pdf: Path) -> None:
    svc, metadata = _make_service()
    src = await svc.render(str(sample_pdf), knowledge_id="k1")
    assert src.status == "rendered"
    rows = await metadata.get_page_analyses(src.source_id)
    assert len(rows) == 2
    for r in rows:
        assert "page_image_b64" in r["data"]
        # PNG magic in base64 starts with iVBOR
        assert r["data"]["page_image_b64"].startswith("iVBOR")


async def test_render_dxf_produces_single_page_image(sample_dxf: Path) -> None:
    svc, metadata = _make_service()
    src = await svc.render(str(sample_dxf), knowledge_id="k1")
    assert src.status == "rendered"
    assert src.metadata["source_format"] == "dxf"
    rows = await metadata.get_page_analyses(src.source_id)
    assert len(rows) == 1
    assert rows[0]["data"]["source_format"] == "dxf"


async def test_render_is_idempotent_via_file_hash(sample_pdf: Path) -> None:
    svc, _ = _make_service()
    src_a = await svc.render(str(sample_pdf), knowledge_id="k1")
    src_b = await svc.render(str(sample_pdf), knowledge_id="k1")
    assert src_a.source_id == src_b.source_id
    assert src_a.file_hash == src_b.file_hash


async def test_render_rejects_unsupported_extension(tmp_path: Path) -> None:
    svc, _ = _make_service()
    bogus = tmp_path / "fake.txt"
    bogus.write_text("hi")
    with pytest.raises(IngestionError, match="(?i)unsupported drawing extension"):
        await svc.render(str(bogus), knowledge_id="k1")


async def test_render_missing_file_raises(tmp_path: Path) -> None:
    svc, _ = _make_service()
    with pytest.raises((FileNotFoundError, IngestionError)):
        await svc.render(str(tmp_path / "nope.pdf"), knowledge_id="k1")
