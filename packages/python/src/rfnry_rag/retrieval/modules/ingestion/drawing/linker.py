"""Deterministic cross-sheet linker for drawings.

Three passes, all deterministic, zero LLM calls:
1. pair_off_page_connectors - exact tag ("/A2") across pages -> one DetectedConnection per consecutive pair.
2. parse_target_hints - regex "sheet N (zone XN)" inside target_hint -> resolve to the named target page and pair.
3. merge_fuzzy_labels - RapidFuzz WRatio between component labels across pages, above config.fuzzy_label_threshold.
"""

from __future__ import annotations

import re
from collections import defaultdict

from rapidfuzz import fuzz

from rfnry_rag.retrieval.common.logging import get_logger
from rfnry_rag.retrieval.modules.ingestion.drawing.config import DrawingIngestionConfig
from rfnry_rag.retrieval.modules.ingestion.drawing.models import (
    DetectedConnection,
    DrawingPageAnalysis,
)

logger = get_logger("drawing/ingestion/linker")

_SHEET_HINT_RE = re.compile(r"sheet\s+(\d+)(?:\s+zone\s+([A-H]\d+))?", re.IGNORECASE)


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
            # Unbound connectors (no on-page anchor) can't form a paired
            # DetectedConnection — skip; they remain available as page metadata.
            if opc.bound_component is None:
                continue
            by_tag[opc.tag].append((pa.page_number, opc.bound_component))

    pairings: list[DetectedConnection] = []
    for tag, occurrences in by_tag.items():
        if len(occurrences) < 2:
            continue
        # Sort by page_number so consecutive pairings are deterministic
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


def parse_target_hints(pages: list[DrawingPageAnalysis], config: DrawingIngestionConfig) -> list[DetectedConnection]:
    """Parse off_page_connectors.target_hint -> resolve target page + pair.

    Heuristic: match regex 'sheet N (zone XN)?' inside target_hint; resolve
    to the target page; pair with any off_page_connector on that page whose
    tag matches (exact or fuzzy-equal).
    """
    pairings: list[DetectedConnection] = []
    page_numbers = {p.page_number: p for p in pages}
    for pa in pages:
        for opc in pa.off_page_connectors:
            if not opc.target_hint:
                continue
            if opc.bound_component is None:
                continue
            m = _SHEET_HINT_RE.search(opc.target_hint)
            if not m:
                continue
            target_page = int(m.group(1))
            target_pa = page_numbers.get(target_page)
            if target_pa is None:
                continue
            for target_opc in target_pa.off_page_connectors:
                if target_opc.bound_component is None:
                    continue
                if target_opc.tag == opc.tag or _tags_similar(target_opc.tag, opc.tag):
                    pairings.append(
                        DetectedConnection(
                            from_component=opc.bound_component,
                            to_component=target_opc.bound_component,
                            net_label=opc.tag,
                            wire_style="signal",
                            properties={
                                "cross_sheet": True,
                                "via": "target_hint",
                                "from_page": pa.page_number,
                                "to_page": target_page,
                            },
                        )
                    )
                    break
    return pairings


def merge_fuzzy_labels(
    pages: list[DrawingPageAnalysis], config: DrawingIngestionConfig
) -> list[tuple[int, str, int, str]]:
    """Return (page_a, component_a, page_b, component_b) tuples to merge.

    Uses RapidFuzz WRatio, threshold from config.fuzzy_label_threshold (fraction
    in [0, 1]; multiplied by 100 for rapidfuzz's 0-100 scale).

    Each component is consumed by at most one merge - the first (earliest
    page, earliest position) match wins.
    """
    merges: list[tuple[int, str, int, str]] = []
    threshold = int(config.fuzzy_label_threshold * 100)
    all_components = [
        (pa.page_number, c.component_id, (c.label or c.component_id)) for pa in pages for c in pa.components
    ]
    seen: set[tuple[int, str]] = set()
    for i, (page_a, id_a, label_a) in enumerate(all_components):
        if (page_a, id_a) in seen:
            continue
        for page_b, id_b, label_b in all_components[i + 1 :]:
            if page_a == page_b:
                continue
            if (page_b, id_b) in seen:
                continue
            if fuzz.WRatio(label_a, label_b) >= threshold:
                seen.add((page_a, id_a))
                seen.add((page_b, id_b))
                merges.append((page_a, id_a, page_b, id_b))
                break
    return merges


def _tags_similar(a: str, b: str, threshold: float = 0.9) -> bool:
    return fuzz.ratio(a, b) >= int(threshold * 100)


def find_unresolved_candidates(
    pages: list[DrawingPageAnalysis],
    deterministic_pairings: list[DetectedConnection],
    fuzzy_merges: list[tuple[int, str, int, str]],
) -> list[tuple[int, str]]:
    """Return (page, component_id) tuples that the deterministic passes didn't resolve.

    A component is "resolved" if it:
    - was consumed by a deterministic pairing (off-page-connector match or target-hint match), or
    - was part of a fuzzy merge.

    Unresolved candidates become the residue for LLM synthesis - when the set
    is empty, skip the LLM call entirely.
    """
    resolved: set[tuple[int, str]] = set()
    for pair in deterministic_pairings:
        props = pair.properties or {}
        fp = props.get("from_page")
        tp = props.get("to_page")
        if fp is not None:
            resolved.add((fp, pair.from_component))
        if tp is not None:
            resolved.add((tp, pair.to_component))
    for pa_a, id_a, pa_b, id_b in fuzzy_merges:
        resolved.add((pa_a, id_a))
        resolved.add((pa_b, id_b))

    candidates: list[tuple[int, str]] = []
    for pa in pages:
        for c in pa.components:
            key = (pa.page_number, c.component_id)
            if key not in resolved:
                candidates.append(key)
    return candidates


def build_digest(pages: list[DrawingPageAnalysis]) -> str:
    """Produce a compact per-page digest for SynthesizeDrawingSet.

    Format:
        Page {n}: domain={d}, components=[(id, class, label), ...]
    """
    lines: list[str] = []
    for pa in pages:
        comps = ", ".join(f"({c.component_id}, {c.symbol_class}, {c.label or ''})" for c in pa.components)
        lines.append(f"Page {pa.page_number}: domain={pa.domain}, components=[{comps}]")
    return "\n".join(lines)


def format_already_linked(
    deterministic_pairings: list[DetectedConnection],
    fuzzy_merges: list[tuple[int, str, int, str]],
) -> str:
    """Pretty-print the already-resolved pairings so SynthesizeDrawingSet can skip them."""
    lines: list[str] = []
    for p in deterministic_pairings:
        props = p.properties or {}
        fp = props.get("from_page", "?")
        tp = props.get("to_page", "?")
        lines.append(f"- page {fp} {p.from_component} <-> page {tp} {p.to_component} (net={p.net_label})")
    for pa_a, id_a, pa_b, id_b in fuzzy_merges:
        lines.append(f"- page {pa_a} {id_a} ~ page {pa_b} {id_b} (fuzzy)")
    return "\n".join(lines) if lines else "(none)"
