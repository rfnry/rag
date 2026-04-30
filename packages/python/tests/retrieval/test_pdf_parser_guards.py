from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rfnry_rag.ingestion.chunk.parsers.pdf import (
    _MAX_PDF_BYTES,
    _MAX_PDF_PAGES,
    PDFParser,
)


def test_pdf_parser_rejects_files_above_size_limit(tmp_path: Path) -> None:
    big = tmp_path / "big.pdf"
    # Avoid writing 500 MiB; stub the stat() size instead.
    big.write_bytes(b"%PDF-1.4\ntiny content")
    with (
        patch.object(Path, "stat", return_value=MagicMock(st_size=_MAX_PDF_BYTES + 1)),
        pytest.raises(ValueError, match="exceeds cap"),
    ):
        PDFParser().parse(str(big))


def test_pdf_parser_rejects_too_many_pages(tmp_path: Path) -> None:
    # Write a real (tiny) file so Path.stat() passes the size check.
    small = tmp_path / "many_pages.pdf"
    small.write_bytes(b"%PDF-1.4\ntiny content")

    fake_doc = MagicMock()
    fake_doc.page_count = _MAX_PDF_PAGES + 1
    fake_doc.__enter__ = MagicMock(return_value=fake_doc)
    fake_doc.__exit__ = MagicMock(return_value=None)

    with (
        patch(
            "rfnry_rag.ingestion.chunk.parsers.pdf.pymupdf.open",
            return_value=fake_doc,
        ),
        pytest.raises(ValueError, match="exceeds cap"),
    ):
        PDFParser().parse(str(small))
