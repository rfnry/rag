import base64
import hashlib
from collections.abc import Iterator
from pathlib import Path

import pymupdf

from rfnry_rag.common.logging import get_logger

logger = get_logger("analyze/ingestion/analyze")


def iter_pdf_page_images(file_path: Path, dpi: int = 300, pages: set[int] | None = None) -> Iterator[dict]:
    """Yield PDF pages as dicts with page metadata and rendered image.

    Keys yielded per page:
    - ``page_number``: 1-based page number
    - ``image_base64``: PNG render of the page as a base64 string
    - ``raw_text``: plain-text extraction via PyMuPDF (may be empty for scanned pages)
    - ``raw_text_char_count``: length of stripped raw_text (0 for scanned/image-only pages)
    - ``has_images``: True when the page contains at least one embedded raster/vector image
    - ``page_hash``: SHA-256 (first 32 hex chars) of the rendered PNG bytes

    Streaming the pages keeps only one rendered image in memory at a time,
    instead of materializing every page before any is processed.

    ``raw_text_char_count`` and ``has_images`` are used by the text-density
    pre-filter in ``_analyze_pdf_with_cache`` to skip vision LLM calls for
    pages that are text-only and image-free."""
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
            raw_text = page.get_text()
            raw_text_char_count = len(raw_text.strip())
            has_images = bool(page.get_images(full=False))
            pix = page.get_pixmap(matrix=matrix)
            png_bytes = pix.tobytes("png")
            page_hash = hashlib.sha256(png_bytes).hexdigest()[:32]
            b64 = base64.standard_b64encode(png_bytes).decode("utf-8")
            emitted += 1
            yield {
                "page_number": page_1based,
                "image_base64": b64,
                "raw_text": raw_text,
                "raw_text_char_count": raw_text_char_count,
                "has_images": has_images,
                "page_hash": page_hash,
            }
        logger.info("streamed pdf as %d page images at %d dpi", emitted, dpi)
    finally:
        doc.close()
