"""Per-page vision failures soft-skip with a note instead of aborting the source.

Pins the contract that:

- One bad page among many leaves the others ingested and produces a
  ``vision:warn:page_<n>:...`` note.
- All pages failing raises ``IngestionError`` (no source created).
- A clean run produces no notes.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from baml_py import errors as baml_errors

from rfnry_rag.exceptions import IngestionError
from rfnry_rag.ingestion.analyze.service import AnalyzedIngestionService


def _baml_result(page_num: int) -> SimpleNamespace:
    return SimpleNamespace(
        description=f"page {page_num}",
        entities=[],
        tables=[],
        annotations=[],
        page_type="diagram",
    )


def _pages(n: int) -> list[dict]:
    return [
        {
            "page_number": i + 1,
            "image_base64": "aW1n",
            "raw_text": "short",
            "raw_text_char_count": 5,
            "has_images": True,
            "page_hash": f"h{i + 1}",
        }
        for i in range(n)
    ]


class _FakeEmbeddings:
    @property
    def model(self) -> str:
        return "fake"

    async def embed(self, texts):
        return [[0.1] * 4 for _ in texts]

    async def embedding_dimension(self) -> int:
        return 4


class _FakeVectorStore:
    async def initialize(self, dim: int) -> None:
        pass

    async def upsert(self, points) -> None:
        pass


def _make_service():
    metadata_store = AsyncMock()
    metadata_store.find_by_hash = AsyncMock(return_value=None)
    metadata_store.create_source = AsyncMock()
    metadata_store.upsert_page_analyses = AsyncMock()
    metadata_store.get_page_analyses_by_hash = AsyncMock(return_value={})

    svc = AnalyzedIngestionService(
        embeddings=_FakeEmbeddings(),
        vector_store=_FakeVectorStore(),
        metadata_store=metadata_store,
        embedding_model_name="fake",
        vision=MagicMock(),
        analyze_text_skip_threshold_chars=0,
    )
    svc._registry = MagicMock()
    return svc


async def test_vision_one_bad_page_among_many_soft_skips(tmp_path) -> None:
    svc = _make_service()
    pages = _pages(5)

    captured_pages: list[int] = []

    async def selective_baml(image, *, baml_options=None):
        # The image bytes don't carry the page number; we go round-robin over the
        # call sequence. Page 3 throws — page index ordering matches gather order.
        idx = len(captured_pages)
        captured_pages.append(idx + 1)
        if captured_pages[-1] == 3:
            raise RuntimeError("rate limited on page 3")
        return _baml_result(captured_pages[-1])

    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4")

    with (
        patch(
            "rfnry_rag.ingestion.analyze.service.iter_pdf_page_images",
            return_value=iter(pages),
        ),
        patch(
            "rfnry_rag.ingestion.analyze.service.compute_file_hash",
            return_value="hash",
        ),
        patch(
            "rfnry_rag.ingestion.analyze.service.asyncio.to_thread",
            new_callable=AsyncMock,
            side_effect=lambda fn, *args: fn(*args),
        ),
        patch("rfnry_rag.baml.baml_client.async_client.b") as mock_b,
    ):
        mock_b.AnalyzePage = AsyncMock(side_effect=selective_baml)
        source = await svc.analyze(file_path=pdf)

    notes = source.metadata.get("ingestion_notes", [])
    assert any(n.startswith("vision:warn:page_3:") for n in notes), notes
    upsert_call = svc._metadata_store.upsert_page_analyses.await_args
    rows = upsert_call.args[1]
    assert len(rows) == 4


async def test_vision_all_pages_fail_raises(tmp_path) -> None:
    svc = _make_service()
    pages = _pages(3)

    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4")

    with (
        patch(
            "rfnry_rag.ingestion.analyze.service.iter_pdf_page_images",
            return_value=iter(pages),
        ),
        patch(
            "rfnry_rag.ingestion.analyze.service.compute_file_hash",
            return_value="hash2",
        ),
        patch(
            "rfnry_rag.ingestion.analyze.service.asyncio.to_thread",
            new_callable=AsyncMock,
            side_effect=lambda fn, *args: fn(*args),
        ),
        patch("rfnry_rag.baml.baml_client.async_client.b") as mock_b,
    ):
        mock_b.AnalyzePage = AsyncMock(side_effect=RuntimeError("everything broken"))
        with pytest.raises(IngestionError, match="all pages"):
            await svc.analyze(file_path=pdf)

    svc._metadata_store.create_source.assert_not_called()


async def test_vision_invalid_baml_output_writes_invalid_output_note(tmp_path) -> None:
    svc = _make_service()
    pages = _pages(2)

    call_count = {"n": 0}

    async def selective_baml(image, *, baml_options=None):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise baml_errors.BamlValidationError(
                "AnalyzePage", "missing required field", "", "field 'X' missing"
            )
        return _baml_result(call_count["n"])

    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4")

    with (
        patch(
            "rfnry_rag.ingestion.analyze.service.iter_pdf_page_images",
            return_value=iter(pages),
        ),
        patch(
            "rfnry_rag.ingestion.analyze.service.compute_file_hash",
            return_value="hash3",
        ),
        patch(
            "rfnry_rag.ingestion.analyze.service.asyncio.to_thread",
            new_callable=AsyncMock,
            side_effect=lambda fn, *args: fn(*args),
        ),
        patch("rfnry_rag.baml.baml_client.async_client.b") as mock_b,
    ):
        mock_b.AnalyzePage = AsyncMock(side_effect=selective_baml)
        source = await svc.analyze(file_path=pdf)

    notes = source.metadata.get("ingestion_notes", [])
    assert any("invalid_output" in n for n in notes), notes


async def test_vision_clean_no_notes(tmp_path) -> None:
    svc = _make_service()
    pages = _pages(3)

    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4")

    with (
        patch(
            "rfnry_rag.ingestion.analyze.service.iter_pdf_page_images",
            return_value=iter(pages),
        ),
        patch(
            "rfnry_rag.ingestion.analyze.service.compute_file_hash",
            return_value="hash4",
        ),
        patch(
            "rfnry_rag.ingestion.analyze.service.asyncio.to_thread",
            new_callable=AsyncMock,
            side_effect=lambda fn, *args: fn(*args),
        ),
        patch("rfnry_rag.baml.baml_client.async_client.b") as mock_b,
    ):
        mock_b.AnalyzePage = AsyncMock(side_effect=lambda image, **kw: _baml_result(1))
        source = await svc.analyze(file_path=pdf)

    assert source.metadata.get("ingestion_notes", []) == []
    assert source.fully_ingested is True
