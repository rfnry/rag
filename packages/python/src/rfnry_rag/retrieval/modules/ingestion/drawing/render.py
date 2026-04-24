"""Render page images from PDF (PyMuPDF) or DXF (ezdxf)."""
from __future__ import annotations

import base64
import hashlib
from collections.abc import Iterator
from io import BytesIO
from pathlib import Path

from rfnry_rag.retrieval.modules.ingestion.analyze.pdf_splitter import iter_pdf_page_images


def render_pdf_pages(file_path: Path, dpi: int) -> Iterator[dict]:
    """Yield page dicts {page_number, image_base64, page_hash, raw_text, raw_text_char_count, has_images}.

    Thin wrapper over iter_pdf_page_images for symmetry with render_dxf.
    """
    yield from iter_pdf_page_images(file_path, dpi=dpi)


def render_dxf(file_path: Path, dpi: int) -> dict:
    """Render a DXF modelspace to a single PNG; return a splitter-shaped dict.

    Paperspace layouts are deferred to Phase D.
    """
    import ezdxf
    import matplotlib

    # Force non-interactive Agg backend; DXF rendering runs off the main thread
    # (via asyncio.to_thread), which is incompatible with GUI backends like TkAgg.
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from ezdxf.addons.drawing import Frontend, RenderContext
    from ezdxf.addons.drawing.matplotlib import MatplotlibBackend

    doc = ezdxf.readfile(str(file_path))
    msp = doc.modelspace()

    fig, ax = plt.subplots(figsize=(16, 12))
    try:
        ctx = RenderContext(doc)
        backend = MatplotlibBackend(ax)
        frontend = Frontend(ctx, backend)
        frontend.draw_layout(msp, finalize=True)

        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    finally:
        plt.close(fig)

    png_bytes = buf.getvalue()
    return {
        "page_number": 1,
        "image_base64": base64.standard_b64encode(png_bytes).decode("utf-8"),
        "page_hash": hashlib.sha256(png_bytes).hexdigest()[:32],
        "raw_text": "",  # DXF text extraction in C6 via direct entity parse
        "raw_text_char_count": 0,
        "has_images": True,
        "source_format": "dxf",
    }
