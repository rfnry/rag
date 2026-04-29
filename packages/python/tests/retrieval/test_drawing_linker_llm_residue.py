"""Link phase LLM residue pass: SynthesizeDrawingSet only on ambiguity."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

from rfnry_rag.retrieval.common.language_model import LanguageModelClient, LanguageModelProvider
from rfnry_rag.retrieval.common.models import Source
from rfnry_rag.retrieval.modules.ingestion.drawing.config import DrawingIngestionConfig
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


def _lm_config(multi_page_linking: bool = True) -> DrawingIngestionConfig:
    lm = LanguageModelClient(
        provider=LanguageModelProvider(provider="openai", api_key="sk-test", model="gpt-4o"),
    )
    return DrawingIngestionConfig(enabled=True, lm_client=lm, multi_page_linking=multi_page_linking)


def _make_service(
    metadata: _InMemoryMetadataStore,
    config: DrawingIngestionConfig | None = None,
) -> DrawingIngestionService:
    return DrawingIngestionService(
        config=config or _lm_config(),
        embeddings=SimpleNamespace(),  # type: ignore[arg-type]
        vector_store=SimpleNamespace(),  # type: ignore[arg-type]
        metadata_store=metadata,  # type: ignore[arg-type]
        embedding_model_name="test-embed",
    )


async def _seed_extracted(metadata: _InMemoryMetadataStore, pages: list[DrawingPageAnalysis]) -> Source:
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


def _fake_merge(
    page_a: int,
    component_a: str,
    page_b: int,
    component_b: str,
    confidence: float,
    rationale: str = "llm says so",
) -> SimpleNamespace:
    return SimpleNamespace(
        page_a=page_a, component_a=component_a,
        page_b=page_b, component_b=component_b,
        confidence=confidence, rationale=rationale,
    )


def _fake_synthesis(merges: list | None = None, xrefs: list | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        ambiguous_component_merges=merges or [],
        narrative_cross_references=xrefs or [],
        document_summary="synthesis stub",
    )


async def test_no_llm_call_when_no_unresolved_candidates() -> None:
    """All pages cleanly linked deterministically -> no LLM call."""
    p1 = _dp(1, [_dc("R1")], [OffPageConnector(tag="/A2", bound_component="R1")])
    p3 = _dp(3, [_dc("C5", "capacitor")], [OffPageConnector(tag="/A2", bound_component="C5")])

    metadata = _InMemoryMetadataStore()
    svc = _make_service(metadata)
    src = await _seed_extracted(metadata, [p1, p3])

    mock_synth = AsyncMock(return_value=_fake_synthesis())
    with patch(
        "rfnry_rag.retrieval.modules.ingestion.drawing.service.b.SynthesizeDrawingSet",
        mock_synth,
    ):
        src = await svc.link(src.source_id)

    # Deterministic pass already resolved /A2 across pages 1+3.
    # There are no un-paired off_page_connectors and no unlabeled components
    # with conflicting fuzzy matches -> no LLM call.
    assert mock_synth.call_count == 0
    assert src.metadata["drawing_linking"]["llm_residue"] == []


async def test_llm_call_fires_when_ambiguous_labels_exist() -> None:
    """Two pages with similar-but-not-identical labels -> LLM residue pass triggers."""
    p1 = _dp(1, [_dc("v1", "valve_ball", label="V-101 Feed valve")])
    p2 = _dp(2, [_dc("v2", "valve_ball", label="V-101-A feeder valve")])

    metadata = _InMemoryMetadataStore()
    svc = _make_service(metadata)
    src = await _seed_extracted(metadata, [p1, p2])

    mock_synth = AsyncMock(return_value=_fake_synthesis(
        merges=[_fake_merge(1, "v1", 2, "v2", confidence=0.85)],
    ))
    with patch(
        "rfnry_rag.retrieval.modules.ingestion.drawing.service.b.SynthesizeDrawingSet",
        mock_synth,
    ):
        src = await svc.link(src.source_id)

    assert mock_synth.call_count == 1
    residue = src.metadata["drawing_linking"]["llm_residue"]
    assert len(residue) == 1
    assert residue[0]["component_a"] == "v1"
    assert residue[0]["component_b"] == "v2"
    assert residue[0]["confidence"] == 0.85


async def test_llm_merges_below_confidence_dropped() -> None:
    """Merges with confidence < 0.5 must not make it into llm_residue."""
    p1 = _dp(1, [_dc("v1", "valve_ball", label="V-101 Feed valve")])
    p2 = _dp(2, [_dc("v2", "valve_ball", label="V-102 outlet")])

    metadata = _InMemoryMetadataStore()
    svc = _make_service(metadata)
    src = await _seed_extracted(metadata, [p1, p2])

    mock_synth = AsyncMock(return_value=_fake_synthesis(
        merges=[
            _fake_merge(1, "v1", 2, "v2", confidence=0.30),  # below threshold
            _fake_merge(1, "v1", 2, "v2", confidence=0.95),  # keep
        ],
    ))
    with patch(
        "rfnry_rag.retrieval.modules.ingestion.drawing.service.b.SynthesizeDrawingSet",
        mock_synth,
    ):
        src = await svc.link(src.source_id)

    residue = src.metadata["drawing_linking"]["llm_residue"]
    assert len(residue) == 1
    assert residue[0]["confidence"] == 0.95


async def test_multi_page_linking_disabled_skips_llm() -> None:
    """Even with unresolved candidates, config.multi_page_linking=False skips the LLM call."""
    p1 = _dp(1, [_dc("v1", "valve_ball", label="V-101")])
    p2 = _dp(2, [_dc("v2", "valve_ball", label="V-101A")])

    metadata = _InMemoryMetadataStore()
    svc = _make_service(metadata, config=_lm_config(multi_page_linking=False))
    src = await _seed_extracted(metadata, [p1, p2])

    mock_synth = AsyncMock(return_value=_fake_synthesis())
    with patch(
        "rfnry_rag.retrieval.modules.ingestion.drawing.service.b.SynthesizeDrawingSet",
        mock_synth,
    ):
        src = await svc.link(src.source_id)

    assert mock_synth.call_count == 0
    assert src.metadata["drawing_linking"]["llm_residue"] == []


async def test_llm_call_skipped_when_no_registry() -> None:
    """No lm_client -> no ClientRegistry -> silent-skip the LLM residue pass."""
    p1 = _dp(1, [_dc("v1", label="V-101")])
    p2 = _dp(2, [_dc("v2", label="V-101-A")])
    cfg = DrawingIngestionConfig(enabled=True)   # no lm_client
    metadata = _InMemoryMetadataStore()
    svc = _make_service(metadata, config=cfg)
    src = await _seed_extracted(metadata, [p1, p2])

    mock_synth = AsyncMock(return_value=_fake_synthesis())
    with patch(
        "rfnry_rag.retrieval.modules.ingestion.drawing.service.b.SynthesizeDrawingSet",
        mock_synth,
    ):
        src = await svc.link(src.source_id)

    assert mock_synth.call_count == 0
    assert src.metadata["drawing_linking"]["llm_residue"] == []
