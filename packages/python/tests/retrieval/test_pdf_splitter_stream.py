"""PDF splitter streaming test — the iterator must yield pages lazily so only
one rendered image lives in memory at a time. We use pymupdf on a tiny
synthetic PDF and assert the returned object is a generator (not a list)."""

import types

import pymupdf
import pytest

from rfnry_rag.ingestion.analyze.pdf_splitter import iter_pdf_page_images


@pytest.fixture
def tiny_pdf(tmp_path):
    pdf_path = tmp_path / "t.pdf"
    doc = pymupdf.open()
    for i in range(3):
        page = doc.new_page()
        page.insert_text((72, 72), f"page {i + 1}")
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


def test_iter_pdf_page_images_returns_generator(tiny_pdf) -> None:
    """Must be a generator, not a materialized list — that's the whole point of
    the refactor."""
    result = iter_pdf_page_images(tiny_pdf, dpi=72)
    assert isinstance(result, types.GeneratorType)


def test_iter_pdf_page_images_yields_pages_in_order(tiny_pdf) -> None:
    pages = list(iter_pdf_page_images(tiny_pdf, dpi=72))
    assert [p["page_number"] for p in pages] == [1, 2, 3]
    for p in pages:
        assert p["image_base64"]  # non-empty base64 string


def test_iter_pdf_page_images_filters_by_page_range(tiny_pdf) -> None:
    pages = list(iter_pdf_page_images(tiny_pdf, dpi=72, pages={2}))
    assert [p["page_number"] for p in pages] == [2]


def test_iter_pdf_page_images_file_missing(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        list(iter_pdf_page_images(tmp_path / "nope.pdf"))
