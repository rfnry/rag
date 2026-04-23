import base64
from collections.abc import Iterator
from pathlib import Path

import pymupdf

from rfnry_rag.retrieval.common.logging import get_logger

logger = get_logger("analyze/ingestion/analyze")


def iter_pdf_page_images(
    file_path: Path, dpi: int = 300, pages: set[int] | None = None
) -> Iterator[dict]:
    """Yield PDF pages as {"page_number", "image_base64"} one at a time.

    Streaming the pages keeps only one rendered image in memory at a time,
    instead of materializing every page before any is processed."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    doc = pymupdf.open(str(file_path))
    zoom = dpi / 72.0
    matrix = pymupdf.Matrix(zoom, zoom)
    emitted = 0
    try:
        for page_num in range(len(doc)):
            page_1based = page_num + 1
            if pages is not None and page_1based not in pages:
                continue
            page = doc[page_num]
            pix = page.get_pixmap(matrix=matrix)
            png_bytes = pix.tobytes("png")
            b64 = base64.standard_b64encode(png_bytes).decode("utf-8")
            emitted += 1
            yield {"page_number": page_1based, "image_base64": b64}
        logger.info("streamed pdf as %d page images at %d dpi", emitted, dpi)
    finally:
        doc.close()
