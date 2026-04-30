"""Drawing extract phase (PDF): AnalyzeDrawingPage per page, idempotent re-entry."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from rfnry_rag.common.language_model import LanguageModelClient, LanguageModelProvider
from rfnry_rag.retrieval.common.errors import IngestionError
from rfnry_rag.retrieval.modules.ingestion.drawing.config import DrawingIngestionConfig
from rfnry_rag.retrieval.modules.ingestion.drawing.service import DrawingIngestionService


class _InMemoryMetadataStore:
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
        # Merge on (source_id, page_number) key — last-write-wins per page
        existing = {r["page_number"]: r for r in self._pages.get(source_id, [])}
        for r in analyses:
            prior = existing.get(r["page_number"], {})
            merged_data = {**prior.get("data", {}), **r.get("data", {})}
            existing[r["page_number"]] = {**prior, **r, "data": merged_data}
        self._pages[source_id] = list(existing.values())

    async def get_page_analyses(self, source_id: str) -> list[dict]:
        return list(self._pages.get(source_id, []))

    async def get_source(self, source_id: str):
        return self._sources.get(source_id)


def _make_config_with_lm() -> DrawingIngestionConfig:
    # Minimal lm_client so DrawingIngestionService initializes a BAML ClientRegistry.
    lm = LanguageModelClient(
        provider=LanguageModelProvider(provider="openai", api_key="sk-test", model="gpt-4o"),
    )
    return DrawingIngestionConfig(enabled=True, lm_client=lm)


def _make_service(metadata, config=None) -> DrawingIngestionService:
    cfg = config or _make_config_with_lm()
    return DrawingIngestionService(
        config=cfg,
        embeddings=SimpleNamespace(),
        vector_store=SimpleNamespace(),
        metadata_store=metadata,  # type: ignore[arg-type]
        embedding_model_name="test-embed",
    )


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    import pymupdf

    path = tmp_path / "two_page.pdf"
    doc = pymupdf.open()
    for i in range(2):
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 72), f"Page {i + 1}")
    doc.save(path)
    doc.close()
    return path


def _fake_drawing_result():
    """Build a minimal DrawingPageAnalysis-shaped object with BAML attr names."""
    return SimpleNamespace(
        page_number=1,
        sheet_number=None,
        zone_grid=None,
        domain="electrical",
        components=[],
        connections=[],
        off_page_connectors=[],
        title_block=None,
        notes=[],
        page_type="drawing",
    )


async def test_extract_calls_analyze_drawing_page_per_page(sample_pdf: Path) -> None:
    metadata = _InMemoryMetadataStore()
    svc = _make_service(metadata)
    src = await svc.render(str(sample_pdf), knowledge_id="k1")

    mock_analyze = AsyncMock(side_effect=lambda *a, **kw: _fake_drawing_result())
    with patch(
        "rfnry_rag.retrieval.modules.ingestion.drawing.extract_pdf.b.AnalyzeDrawingPage",
        mock_analyze,
    ):
        src = await svc.extract(src.source_id)

    assert src.status == "extracted"
    assert mock_analyze.call_count == 2


async def test_extract_passes_consumer_symbol_library(sample_pdf: Path) -> None:
    metadata = _InMemoryMetadataStore()
    svc = _make_service(metadata)
    src = await svc.render(str(sample_pdf), knowledge_id="k1")

    mock_analyze = AsyncMock(side_effect=lambda *a, **kw: _fake_drawing_result())
    with patch(
        "rfnry_rag.retrieval.modules.ingestion.drawing.extract_pdf.b.AnalyzeDrawingPage",
        mock_analyze,
    ):
        await svc.extract(src.source_id)

    # Call kwargs should contain the serialised symbol_library string
    _, kwargs = mock_analyze.call_args_list[0]
    # Either positional (page_image, domain_hint, symbol_library, off_page_patterns)
    # or kwarg form — tolerate both.
    args_call = mock_analyze.call_args_list[0]
    flat_args = list(args_call.args) + list(args_call.kwargs.values())
    symbol_library_joined = " ".join(str(a) for a in flat_args)
    assert "resistor" in symbol_library_joined
    # off-page patterns appear somewhere in call args
    assert "OPC" in symbol_library_joined or "sheet" in symbol_library_joined.lower()


async def test_extract_reuses_cached_analyses_on_re_entry(sample_pdf: Path) -> None:
    metadata = _InMemoryMetadataStore()
    svc = _make_service(metadata)
    src = await svc.render(str(sample_pdf), knowledge_id="k1")

    mock_analyze = AsyncMock(side_effect=lambda *a, **kw: _fake_drawing_result())
    with patch(
        "rfnry_rag.retrieval.modules.ingestion.drawing.extract_pdf.b.AnalyzeDrawingPage",
        mock_analyze,
    ):
        src = await svc.extract(src.source_id)
        count_after_first = mock_analyze.call_count

        # Second call must be a no-op (status already 'extracted')
        src_again = await svc.extract(src.source_id)
        assert src_again.status == "extracted"
        assert mock_analyze.call_count == count_after_first


async def test_extract_requires_rendered_status(sample_pdf: Path) -> None:
    """Calling extract on a Source that hasn't been rendered yet is a hard error."""
    from rfnry_rag.retrieval.common.models import Source

    metadata = _InMemoryMetadataStore()
    svc = _make_service(metadata)
    # Hand-craft a Source with an unexpected status
    bogus = Source(source_id="s-bogus", status="pending", file_hash="h", knowledge_id="k1")
    await metadata.create_source(bogus)
    with pytest.raises(IngestionError, match="(?i)requires status"):
        await svc.extract("s-bogus")


async def test_extract_unknown_source_id_raises() -> None:
    metadata = _InMemoryMetadataStore()
    svc = _make_service(metadata)
    with pytest.raises(IngestionError, match="(?i)source not found"):
        await svc.extract("does-not-exist")


async def test_extract_respects_analyze_concurrency() -> None:
    """Semaphore must cap concurrent AnalyzeDrawingPage calls at config.analyze_concurrency."""
    import asyncio
    from pathlib import Path as _Path
    from tempfile import mkdtemp

    import pymupdf

    # Build a 4-page PDF
    tmp = _Path(mkdtemp())
    pdf = tmp / "many.pdf"
    doc = pymupdf.open()
    for i in range(4):
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 72), f"P{i + 1}")
    doc.save(pdf)
    doc.close()

    lm = LanguageModelClient(
        provider=LanguageModelProvider(provider="openai", api_key="sk-test", model="gpt-4o"),
    )
    cfg = DrawingIngestionConfig(enabled=True, lm_client=lm, analyze_concurrency=2)
    metadata = _InMemoryMetadataStore()
    svc = _make_service(metadata, config=cfg)
    src = await svc.render(str(pdf), knowledge_id="k1")

    in_flight = 0
    max_in_flight = 0

    async def track(*args, **kwargs):
        nonlocal in_flight, max_in_flight
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return _fake_drawing_result()

    mock_analyze = AsyncMock(side_effect=track)
    with patch(
        "rfnry_rag.retrieval.modules.ingestion.drawing.extract_pdf.b.AnalyzeDrawingPage",
        mock_analyze,
    ):
        await svc.extract(src.source_id)

    assert max_in_flight <= 2
    assert mock_analyze.call_count == 4
