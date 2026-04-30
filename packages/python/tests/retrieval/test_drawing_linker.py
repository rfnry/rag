"""Drawing linker: deterministic exact-tag + regex-target-hint + fuzzy-label merges."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from rfnry_rag.retrieval.common.errors import IngestionError
from rfnry_rag.retrieval.common.models import Source
from rfnry_rag.retrieval.modules.ingestion.drawing.config import DrawingIngestionConfig
from rfnry_rag.retrieval.modules.ingestion.drawing.linker import (
    merge_fuzzy_labels,
    pair_off_page_connectors,
    parse_target_hints,
)
from rfnry_rag.retrieval.modules.ingestion.drawing.models import (
    DetectedComponent,
    DrawingPageAnalysis,
    OffPageConnector,
)
from rfnry_rag.retrieval.modules.ingestion.drawing.service import DrawingIngestionService


def _dc(component_id: str, symbol_class: str = "resistor", label: str | None = None) -> DetectedComponent:
    return DetectedComponent(
        component_id=component_id,
        symbol_class=symbol_class,
        label=label,
        bbox=[0, 0, 10, 10],
        ports=[],
        properties=None,
    )


def _dp(
    page: int,
    components: list[DetectedComponent],
    opcs: list[OffPageConnector] | None = None,
) -> DrawingPageAnalysis:
    return DrawingPageAnalysis(
        page_number=page,
        components=components,
        connections=[],
        off_page_connectors=opcs or [],
        domain="electrical",
        page_type="drawing",
        notes=[],
    )


# ---- pair_off_page_connectors ----


def test_exact_match_off_page_tag_pairs_across_pages() -> None:
    p1 = _dp(1, [_dc("R1")], [OffPageConnector(tag="/A2", bound_component="R1")])
    p3 = _dp(3, [_dc("C5", "capacitor")], [OffPageConnector(tag="/A2", bound_component="C5")])

    pairings = pair_off_page_connectors([p1, p3])
    assert len(pairings) == 1
    p = pairings[0]
    assert (p.from_component, p.to_component) in (("R1", "C5"), ("C5", "R1"))
    assert p.net_label == "/A2"


def test_off_page_tag_three_pages_chains_consecutively() -> None:
    p1 = _dp(1, [_dc("R1")], [OffPageConnector(tag="/A2", bound_component="R1")])
    p3 = _dp(3, [_dc("C5", "capacitor")], [OffPageConnector(tag="/A2", bound_component="C5")])
    p5 = _dp(5, [_dc("L7", "inductor")], [OffPageConnector(tag="/A2", bound_component="L7")])
    pairings = pair_off_page_connectors([p1, p3, p5])
    assert len(pairings) == 2
    # Each pairing carries cross_sheet + from_page/to_page metadata
    for pair in pairings:
        assert pair.properties is not None
        assert pair.properties.get("cross_sheet") is True


def test_single_occurrence_produces_no_pairing() -> None:
    p1 = _dp(1, [_dc("R1")], [OffPageConnector(tag="/A2", bound_component="R1")])
    assert pair_off_page_connectors([p1]) == []


def test_empty_input_returns_no_pairings() -> None:
    assert pair_off_page_connectors([]) == []


# ---- parse_target_hints ----


def test_target_hint_sheet_reference_pairs_across_pages() -> None:
    p1 = _dp(
        1,
        [_dc("R1")],
        [OffPageConnector(tag="#1", bound_component="R1", target_hint="to sheet 3 zone B2")],
    )
    p3 = _dp(
        3,
        [_dc("C5", "capacitor")],
        [OffPageConnector(tag="#1", bound_component="C5", target_hint=None)],
    )
    cfg = DrawingIngestionConfig(enabled=True)
    pairings = parse_target_hints([p1, p3], cfg)
    assert len(pairings) >= 1
    # At least one pairing connects R1 <-> C5
    matched = any((pp.from_component, pp.to_component) in (("R1", "C5"), ("C5", "R1")) for pp in pairings)
    assert matched


def test_target_hint_to_nonexistent_page_is_dropped() -> None:
    p1 = _dp(
        1,
        [_dc("R1")],
        [OffPageConnector(tag="#1", bound_component="R1", target_hint="to sheet 42")],
    )
    cfg = DrawingIngestionConfig(enabled=True)
    assert parse_target_hints([p1], cfg) == []


# ---- merge_fuzzy_labels ----


def test_fuzzy_label_merge_finds_valve_v101_across_pages() -> None:
    p1 = _dp(1, [_dc("v1", "valve_ball", label="V-101")])
    p2 = _dp(2, [_dc("v2", "valve_ball", label="V-101")])
    cfg = DrawingIngestionConfig(enabled=True, fuzzy_label_threshold=0.92)
    merges = merge_fuzzy_labels([p1, p2], cfg)
    assert any(
        {(page_a, id_a, page_b, id_b)} == {(1, "v1", 2, "v2")}
        or (page_a, id_a, page_b, id_b) == (1, "v1", 2, "v2")
        or (page_a, id_a, page_b, id_b) == (2, "v2", 1, "v1")
        for (page_a, id_a, page_b, id_b) in merges
    )


def test_fuzzy_below_threshold_is_not_merged() -> None:
    p1 = _dp(1, [_dc("v1", "valve_ball", label="V-101")])
    p2 = _dp(2, [_dc("v2", "valve_gate", label="totally unrelated text")])
    cfg = DrawingIngestionConfig(enabled=True, fuzzy_label_threshold=0.95)
    assert merge_fuzzy_labels([p1, p2], cfg) == []


def test_fuzzy_same_page_never_merged() -> None:
    p1 = _dp(1, [_dc("v1", label="V-101"), _dc("v2", label="V-101")])
    cfg = DrawingIngestionConfig(enabled=True, fuzzy_label_threshold=0.5)
    assert merge_fuzzy_labels([p1], cfg) == []


# ---- service.link integration (deterministic only; LLM residue is C8) ----


class _InMemoryMetadataStore:
    def __init__(self) -> None:
        self._sources: dict[str, Any] = {}
        self._pages: dict[str, list[dict]] = {}

    async def create_source(self, source: Source) -> None:
        self._sources[source.source_id] = source

    async def update_source(self, source_id: str, **fields: Any) -> None:
        src = self._sources[source_id]
        for k, v in fields.items():
            setattr(src, k, v)

    async def find_by_hash(self, hash_value: str, knowledge_id: str | None) -> None:
        return None

    async def upsert_page_analyses(self, source_id: str, analyses: list[dict]) -> None:
        existing = {r["page_number"]: r for r in self._pages.get(source_id, [])}
        for r in analyses:
            prior = existing.get(r["page_number"], {"page_number": r["page_number"], "data": {}})
            merged_data = {**prior.get("data", {}), **r.get("data", {})}
            existing[r["page_number"]] = {"page_number": r["page_number"], "data": merged_data}
        self._pages[source_id] = list(existing.values())

    async def get_page_analyses(self, source_id: str) -> list[dict]:
        return list(self._pages.get(source_id, []))

    async def get_source(self, source_id: str) -> Any:
        return self._sources.get(source_id)


def _make_service(metadata: _InMemoryMetadataStore) -> DrawingIngestionService:
    cfg = DrawingIngestionConfig(enabled=True)
    return DrawingIngestionService(
        config=cfg,
        embeddings=SimpleNamespace(),  # type: ignore[arg-type]
        vector_store=SimpleNamespace(),  # type: ignore[arg-type]
        metadata_store=metadata,  # type: ignore[arg-type]
        embedding_model_name="test-embed",
    )


async def _seed_extracted_source(metadata: _InMemoryMetadataStore, pages: list[DrawingPageAnalysis]) -> Source:
    src = Source(
        source_id="src-1",
        status="extracted",
        file_hash="abc",
        knowledge_id="k1",
        metadata={"source_format": "pdf"},
    )
    await metadata.create_source(src)
    rows = [{"page_number": p.page_number, "data": {"analysis": p.to_dict()}} for p in pages]
    await metadata.upsert_page_analyses(src.source_id, rows)
    return src


async def test_link_on_wrong_status_raises() -> None:
    metadata = _InMemoryMetadataStore()
    svc = _make_service(metadata)
    bad = Source(source_id="s-1", status="rendered", file_hash="h", knowledge_id="k1", metadata={})
    await metadata.create_source(bad)
    with pytest.raises(IngestionError, match="(?i)requires status"):
        await svc.link("s-1")


async def test_link_missing_source_raises() -> None:
    metadata = _InMemoryMetadataStore()
    svc = _make_service(metadata)
    with pytest.raises(IngestionError, match="(?i)source not found"):
        await svc.link("nope")


async def test_link_populates_drawing_linking_metadata() -> None:
    p1 = _dp(1, [_dc("R1")], [OffPageConnector(tag="/A2", bound_component="R1")])
    p3 = _dp(3, [_dc("C5", "capacitor")], [OffPageConnector(tag="/A2", bound_component="C5")])

    metadata = _InMemoryMetadataStore()
    svc = _make_service(metadata)
    src = await _seed_extracted_source(metadata, [p1, p3])
    src = await svc.link(src.source_id)

    assert src.status == "linked"
    link_payload = src.metadata["drawing_linking"]
    assert "deterministic_pairings" in link_payload
    assert "fuzzy_merges" in link_payload
    assert "llm_residue" in link_payload  # filled by C8
    assert link_payload["llm_residue"] == []
    # The /A2 tag pairing is present
    assert any(d["net_label"] == "/A2" for d in link_payload["deterministic_pairings"])


async def test_link_idempotent_on_already_linked_source() -> None:
    metadata = _InMemoryMetadataStore()
    svc = _make_service(metadata)
    p1 = _dp(1, [_dc("R1")])
    src = await _seed_extracted_source(metadata, [p1])
    # First link
    src1 = await svc.link(src.source_id)
    assert src1.status == "linked"
    # Second link (already-linked) is a no-op
    src2 = await svc.link(src1.source_id)
    assert src2.status == "linked"
