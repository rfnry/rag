"""Drawing render phase: PDF via PyMuPDF, DXF via ezdxf."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from rfnry_rag.retrieval.common.errors import IngestionError
from rfnry_rag.retrieval.modules.ingestion.drawing.config import DrawingIngestionConfig
from rfnry_rag.retrieval.modules.ingestion.drawing.render import render_dxf
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


async def test_render_dxf_produces_per_layout_images(sample_dxf: Path) -> None:
    """A fresh ezdxf doc has Model + the default Layout1 → 2 pages emitted."""
    svc, metadata = _make_service()
    src = await svc.render(str(sample_dxf), knowledge_id="k1")
    assert src.status == "rendered"
    assert src.metadata["source_format"] == "dxf"
    rows = await metadata.get_page_analyses(src.source_id)
    # ezdxf.new() seeds Model + Layout1 deterministically — both render.
    # Empty paperspace still emits a (blank) page; we accept blank pages over
    # silent loss. Asserting the exact count (not >=1) catches a regression
    # where the seeded Layout1 silently fails to render.
    assert len(rows) == 2
    assert rows[0]["page_number"] == 1
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


def test_render_dxf_emits_one_page_per_layout(tmp_path: Path) -> None:
    """Multi-layout DXF: modelspace + 2 paperspace layouts → 3 pages, deterministic order."""
    import ezdxf

    path = tmp_path / "multi_layout.dxf"
    doc = ezdxf.new()
    doc.layouts.new("Layout2")
    # Distinct geometry per layout so the renderer produces visibly different
    # PNGs (matplotlib auto-scales single-line content to identical bitmaps).
    msp = doc.modelspace()
    msp.add_line((0, 0), (10, 0))
    msp.add_line((10, 0), (10, 10))
    ps1 = doc.layouts.get("Layout1")
    ps1.add_line((0, 0), (5, 0))
    ps1.add_line((5, 0), (5, 5))
    ps1.add_line((5, 5), (0, 5))
    ps2 = doc.layouts.get("Layout2")
    for i in range(4):
        ps2.add_line((i * 2, 0), (i * 2, 8))
    doc.saveas(path)

    pages = render_dxf(path, dpi=150)
    assert isinstance(pages, list)
    assert len(pages) == 3
    assert [p["page_number"] for p in pages] == [1, 2, 3]
    # Each page is a distinct image (different layout content → different bytes)
    hashes = {p["page_hash"] for p in pages}
    assert len(hashes) == 3
    for p in pages:
        assert p["source_format"] == "dxf"
        assert p["raw_text"] == ""
        assert p["raw_text_char_count"] == 0
        assert p["has_images"] is False
        assert p["image_base64"].startswith("iVBOR")


def test_render_dxf_modelspace_only_still_works(tmp_path: Path) -> None:
    """Single-layout fixture (only the default Layout1 alongside Model) emits 2 pages."""
    import ezdxf

    path = tmp_path / "msp_only.dxf"
    doc = ezdxf.new()
    msp = doc.modelspace()
    msp.add_line((0, 0), (10, 0))
    doc.saveas(path)

    pages = render_dxf(path, dpi=150)
    # ezdxf.new() seeds Model + Layout1 deterministically — both render.
    # Asserting the exact count (not >=1) catches a regression where
    # the seeded Layout1 silently fails to render.
    assert isinstance(pages, list)
    assert len(pages) == 2
    assert pages[0]["page_number"] == 1
    assert pages[0]["source_format"] == "dxf"


def test_render_dxf_skips_unnamed_layout_alias(tmp_path: Path) -> None:
    """The 'Model' layout (alias of modelspace) must not be double-counted."""
    import ezdxf

    path = tmp_path / "no_double.dxf"
    doc = ezdxf.new()
    doc.layouts.new("Layout2")
    doc.saveas(path)

    pages = render_dxf(path, dpi=150)
    # Expected: 1 modelspace + Layout1 + Layout2 = 3, NOT 4 (no double-counted Model).
    assert len(pages) == 3
    assert [p["page_number"] for p in pages] == [1, 2, 3]
