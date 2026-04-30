"""Deterministic cross-sheet connector pairing.

One pass, deterministic, zero LLM calls: match identical off-page-connector tags
across pages and emit one DetectedConnection per consecutive pair. The graph
store carries every other cross-page reference as edge candidates; cross-sheet
synthesis is the model's job at query time, not the toolkit's at ingest time.
"""

from __future__ import annotations

from collections import defaultdict

from rfnry_rag.common.logging import get_logger
from rfnry_rag.retrieval.modules.ingestion.drawing.models import (
    DetectedConnection,
    DrawingPageAnalysis,
)

logger = get_logger("drawing/ingestion/linker")


def pair_off_page_connectors(
    pages: list[DrawingPageAnalysis],
) -> list[DetectedConnection]:
    """Match identical off_page_connectors.tag across pages.

    For a tag that appears on pages (p1, p3, p5), emit consecutive pairings:
    (p1, p3) and (p3, p5).
    """
    by_tag: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for pa in pages:
        for opc in pa.off_page_connectors:
            if opc.bound_component is None:
                continue
            by_tag[opc.tag].append((pa.page_number, opc.bound_component))

    pairings: list[DetectedConnection] = []
    for tag, occurrences in by_tag.items():
        if len(occurrences) < 2:
            continue
        occurrences.sort(key=lambda t: t[0])
        for i in range(len(occurrences) - 1):
            p_from, c_from = occurrences[i]
            p_to, c_to = occurrences[i + 1]
            pairings.append(
                DetectedConnection(
                    from_component=c_from,
                    to_component=c_to,
                    net_label=tag,
                    wire_style="signal",
                    properties={
                        "cross_sheet": True,
                        "from_page": p_from,
                        "to_page": p_to,
                    },
                )
            )
    return pairings
