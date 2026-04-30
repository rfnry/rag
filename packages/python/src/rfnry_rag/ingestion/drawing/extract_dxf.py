"""Parse DXF entities into DrawingPageAnalysis without any vision LLM."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

from rfnry_rag.common.logging import get_logger
from rfnry_rag.ingestion.drawing.config import DrawingIngestionConfig
from rfnry_rag.ingestion.drawing.models import (
    DetectedComponent,
    DetectedConnection,
    DrawingPageAnalysis,
    OffPageConnector,
)
from rfnry_rag.ingestion.drawing.render import _iter_renderable_layouts

logger = get_logger("drawing/ingestion/extract_dxf")

# Absolute tolerance (in modelspace units) for matching a wire endpoint to an
# INSERT bbox. Kept small to avoid false-positive connections on busy drawings;
# can be promoted to a configurable knob if real-world drawings require it.
_CONNECTION_TOL = 2


def _bbox_of_block_insert(insert: Any) -> list[int]:
    """Approximate axis-aligned bbox of an INSERT entity in integer modelspace coords.

    We don't try to rotate / scale precisely — MVP heuristic based on the
    referenced block's raw entity extents plus the insert's translation.
    """
    xmin = ymin = float("inf")
    xmax = ymax = float("-inf")
    try:
        blk = insert.block()  # ezdxf returns the referenced block table entry
        if blk is not None:
            for e in blk:
                for attr in ("start", "end", "center", "insert"):
                    pt = getattr(getattr(e, "dxf", None), attr, None)
                    if pt is None:
                        continue
                    x, y = float(pt.x), float(pt.y)
                    xmin, ymin = min(xmin, x), min(ymin, y)
                    xmax, ymax = max(xmax, x), max(ymax, y)
    except Exception:
        pass
    if xmin == float("inf"):
        # Unknown extent — give it a small bbox at the insertion point
        ox, oy = float(insert.dxf.insert.x), float(insert.dxf.insert.y)
        return [int(ox), int(oy), 1, 1]
    ox, oy = float(insert.dxf.insert.x), float(insert.dxf.insert.y)
    return [int(ox + xmin), int(oy + ymin), int(xmax - xmin), int(ymax - ymin)]


def _classify_block_name(block_name: str, symbol_library: dict[str, list[str]]) -> tuple[str, str]:
    """Return (domain, symbol_class).

    Matching order (exact > substring either direction). Falls back to
    ("mixed", block_name.lower()) if no entry matches.
    """
    lower = block_name.lower()
    # Pass 1: exact match
    for domain, symbols in symbol_library.items():
        for sym in symbols:
            if sym == lower:
                return domain, sym
    # Pass 2: substring either direction
    for domain, symbols in symbol_library.items():
        for sym in symbols:
            if sym in lower or lower in sym:
                return domain, sym
    return "mixed", lower


def _find_component_at(x: float, y: float, components: list[DetectedComponent]) -> DetectedComponent | None:
    """Locate the component whose bbox contains (x, y).

    Uses a small absolute tolerance (`_CONNECTION_TOL`) so wire endpoints
    terminating just outside a block's bbox still register as connected,
    without inviting false positives on busy drawings.
    """
    for c in components:
        bx, by, bw, bh = c.bbox
        if (bx - _CONNECTION_TOL) <= x <= (bx + bw + _CONNECTION_TOL) and (by - _CONNECTION_TOL) <= y <= (
            by + bh + _CONNECTION_TOL
        ):
            return c
    return None


def _infer_domain(components: list[DetectedComponent]) -> str:
    if not components:
        return "mixed"
    counter: Counter[str] = Counter()
    for c in components:
        props = c.properties or {}
        counter[props.get("domain", "mixed")] += 1
    return counter.most_common(1)[0][0]


def _extract_off_page_connectors(
    layout: Any, components: list[DetectedComponent], patterns: list[str]
) -> list[OffPageConnector]:
    """Scan a layout's TEXT + MTEXT for off-page-connector tags.

    First-match-wins across the configured regex list; entities with no match
    are ignored (label/annotation text is normal noise). MTEXT formatting
    codes are stripped via plain_text(); a corrupt MTEXT is skipped rather
    than aborting the parse so one bad entity can't sink an entire sheet.
    """
    compiled = [re.compile(p) for p in patterns]
    out: list[OffPageConnector] = []
    for e in layout.query("TEXT MTEXT"):
        if e.dxftype() == "MTEXT":
            try:
                payload = e.plain_text(split=False)
            except Exception:
                logger.debug(
                    "[drawing/extract_dxf] skipping corrupt MTEXT handle=%s",
                    getattr(e.dxf, "handle", "?"),
                )
                continue
        else:
            payload = e.dxf.text
        if not payload:
            continue

        match = None
        for cre in compiled:
            match = cre.search(payload)
            if match is not None:
                break
        if match is None:
            continue

        x, y = float(e.dxf.insert.x), float(e.dxf.insert.y)
        bound = _find_component_at(x, y, components)
        tag = match.group(0)
        out.append(
            OffPageConnector(
                tag=tag,
                bound_component=bound.component_id if bound is not None else None,
                target_hint=tag,
            )
        )
    return out


def _analyse_layout(layout: Any, page_number: int, config: DrawingIngestionConfig) -> DrawingPageAnalysis:
    """Walk one layout's INSERT/LINE/TEXT/MTEXT entities into a DrawingPageAnalysis."""
    symbol_library = config.symbol_library or {}

    components: list[DetectedComponent] = []
    for e in layout.query("INSERT"):
        block_name = e.dxf.name
        domain, symbol_class = _classify_block_name(block_name, symbol_library)
        components.append(
            DetectedComponent(
                component_id=str(e.dxf.handle),
                symbol_class=symbol_class,
                label=None,
                bbox=_bbox_of_block_insert(e),
                ports=[],
                properties={
                    "block_name": block_name,
                    "layer": e.dxf.layer,
                    "domain": domain,
                },
            )
        )

    connections: list[DetectedConnection] = []
    for line in layout.query("LINE"):
        x1, y1 = float(line.dxf.start.x), float(line.dxf.start.y)
        x2, y2 = float(line.dxf.end.x), float(line.dxf.end.y)
        src_c = _find_component_at(x1, y1, components)
        tgt_c = _find_component_at(x2, y2, components)
        if src_c and tgt_c and src_c.component_id != tgt_c.component_id:
            connections.append(
                DetectedConnection(
                    from_component=src_c.component_id,
                    to_component=tgt_c.component_id,
                    wire_style="solid",
                )
            )

    off_page_connectors = _extract_off_page_connectors(layout, components, config.off_page_connector_patterns or [])

    return DrawingPageAnalysis(
        page_number=page_number,
        components=components,
        connections=connections,
        off_page_connectors=off_page_connectors,
        domain=_infer_domain(components),
        page_type="drawing",
        notes=[],
    )


def extract_dxf_analysis(file_path: Path, config: DrawingIngestionConfig) -> list[DrawingPageAnalysis]:
    """Parse every layout in a DXF into a list of DrawingPageAnalysis (one per page).

    Mirrors render_dxf's page split: modelspace is page 1, then paperspace
    layouts in DXF tab order (Model alias skipped). INSERT, LINE, TEXT, MTEXT
    are scanned per-layout so multi-sheet drawings expose every layout's
    components, wires, and off-page tags.
    """
    import ezdxf

    doc = ezdxf.readfile(str(file_path))

    # Single source of truth for layout enumeration — render.py uses the same
    # helper, so renderer + extractor stay aligned for service.py's per-page
    # merge by page_number. Drift here would silently misalign analysis with
    # the rendered image.
    analyses: list[DrawingPageAnalysis] = [
        _analyse_layout(layout, idx, config) for idx, layout in enumerate(_iter_renderable_layouts(doc), start=1)
    ]

    logger.info(
        "[drawing/extract_dxf] pages=%d components=%d connections=%d off_page=%d",
        len(analyses),
        sum(len(a.components) for a in analyses),
        sum(len(a.connections) for a in analyses),
        sum(len(a.off_page_connectors) for a in analyses),
    )
    return analyses
