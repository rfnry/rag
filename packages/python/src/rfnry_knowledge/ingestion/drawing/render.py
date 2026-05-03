"""Render page images from PDF (PyMuPDF) or DXF (ezdxf)."""

from __future__ import annotations

import base64
import hashlib
from collections.abc import Iterator
from io import BytesIO
from pathlib import Path
from typing import Any

from rfnry_knowledge.ingestion.analyze.pdf_splitter import iter_pdf_page_images


def render_pdf_pages(file_path: Path, dpi: int) -> Iterator[dict]:
    """Yield page dicts {page_number, image_base64, page_hash, raw_text, raw_text_char_count, has_images}.

    Thin wrapper over iter_pdf_page_images for symmetry with render_dxf.
    """
    yield from iter_pdf_page_images(file_path, dpi=dpi)


def _iter_renderable_layouts(doc: Any) -> list[Any]:
    """Modelspace first, then paperspace layouts in DXF tab order.

    Skips the 'Model' alias from `names_in_taborder()` because it points to
    the same Modelspace instance we already prepended.
    """
    layouts: list[Any] = [doc.modelspace()]
    for name in doc.layouts.names_in_taborder():
        if name.lower() == "model":
            continue
        layouts.append(doc.layouts.get(name))
    return layouts


def render_dxf(file_path: Path, dpi: int) -> list[dict]:
    """Render every layout (modelspace + paperspace) as a separate page.

    Iterates `doc.layouts.names_in_taborder()` (skipping the Model alias) and
    emits one splitter-shaped page dict per layout. Modelspace is page 1; the
    remaining pages follow DXF tab order. An empty paperspace layout still
    renders as a blank page — we favour blank-page-on-empty over silent
    content loss for multi-sheet drawings.
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
    ctx = RenderContext(doc)
    pages: list[dict] = []

    for idx, layout in enumerate(_iter_renderable_layouts(doc), start=1):
        fig, ax = plt.subplots(figsize=(16, 12))
        try:
            backend = MatplotlibBackend(ax)
            frontend = Frontend(ctx, backend)
            frontend.draw_layout(layout, finalize=True)

            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        finally:
            # Close per-iteration to bound matplotlib's figure cache when a
            # drawing carries many layouts.
            plt.close(fig)

        png_bytes = buf.getvalue()
        pages.append(
            {
                "page_number": idx,
                "image_base64": base64.standard_b64encode(png_bytes).decode("utf-8"),
                "page_hash": hashlib.sha256(png_bytes).hexdigest()[:32],
                "raw_text": "",
                "raw_text_char_count": 0,
                "has_images": False,
                "source_format": "dxf",
            }
        )

    return pages
