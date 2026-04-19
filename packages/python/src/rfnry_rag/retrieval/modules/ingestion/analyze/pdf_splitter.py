import base64
from pathlib import Path

import pymupdf

from rfnry_rag.retrieval.common.logging import get_logger

logger = get_logger("analyze/ingestion/analyze")


def split_pdf_to_images(file_path: Path, dpi: int = 300, pages: set[int] | None = None) -> list[dict]:
    """Split PDF into per-page PNG images as base64.

    Returns list of {"page_number": int, "image_base64": str}.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    doc = pymupdf.open(str(file_path))
    images = []
    zoom = dpi / 72.0
    matrix = pymupdf.Matrix(zoom, zoom)
    try:
        for page_num in range(len(doc)):
            page_1based = page_num + 1
            if pages is not None and page_1based not in pages:
                continue
            page = doc[page_num]
            pix = page.get_pixmap(matrix=matrix)
            png_bytes = pix.tobytes("png")
            b64 = base64.standard_b64encode(png_bytes).decode("utf-8")
            images.append({"page_number": page_1based, "image_base64": b64})
        logger.info("split pdf into %d page images at %d dpi", len(images), dpi)
    finally:
        doc.close()
    return images
