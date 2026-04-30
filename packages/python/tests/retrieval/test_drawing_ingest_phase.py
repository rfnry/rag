"""Drawing ingest phase: component vectors + graph writes, idempotent on completed."""

from __future__ import annotations

from typing import Any

import pytest

from rfnry_rag.retrieval.common.errors import IngestionError
from rfnry_rag.retrieval.common.models import Source, VectorPoint
from rfnry_rag.retrieval.modules.ingestion.drawing.config import DrawingIngestionConfig
from rfnry_rag.retrieval.modules.ingestion.drawing.models import (
    DetectedComponent,
    DetectedConnection,
    DrawingPageAnalysis,
)
from rfnry_rag.retrieval.modules.ingestion.drawing.service import DrawingIngestionService


class _InMemoryMetadataStore:
    def __init__(self) -> None:
        self._sources: dict[str, Any] = {}
        self._pages: dict[str, list[dict]] = {}

    async def create_source(self, source) -> None:
        self._sources[source.source_id] = source

    async def update_source(self, source_id, **fields) -> None:
        src = self._sources[source_id]
        for k, v in fields.items():
            setattr(src, k, v)

    async def find_by_hash(self, *a, **k):
        return None

    async def upsert_page_analyses(self, source_id, analyses):
        existing = {r["page_number"]: r for r in self._pages.get(source_id, [])}
        for r in analyses:
            prior = existing.get(r["page_number"], {"page_number": r["page_number"], "data": {}})
            merged_data = {**prior.get("data", {}), **r.get("data", {})}
            existing[r["page_number"]] = {"page_number": r["page_number"], "data": merged_data}
        self._pages[source_id] = list(existing.values())

    async def get_page_analyses(self, source_id):
        return list(self._pages.get(source_id, []))

    async def get_source(self, source_id):
        return self._sources.get(source_id)


class _RecordingVectorStore:
    def __init__(self) -> None:
        self.init_calls: list[int] = []
        self.upserted: list[VectorPoint] = []

    async def initialize(self, vector_size: int) -> None:
        self.init_calls.append(vector_size)

    async def upsert(self, points):
        self.upserted.extend(points)


class _FakeEmbeddings:
    """Deterministic fake: each text becomes a 4-dim vector of its character count ratios."""

    async def embed(self, texts):  # BaseEmbeddings protocol (used by embed_batched)
        return [[float(len(t)) / 10.0, 1.0, 0.0, 0.0] for t in texts]

    async def embedding_dimension(self) -> int:
        return 4


class _RecordingGraphStore:
    def __init__(self) -> None:
        self.entities_calls: list[tuple[str, str | None, list]] = []
        self.relations_calls: list[tuple[str, list]] = []

    async def add_entities(self, source_id, knowledge_id, entities):
        self.entities_calls.append((source_id, knowledge_id, list(entities)))

    async def add_relations(self, source_id, relations):
        self.relations_calls.append((source_id, list(relations)))


def _cfg(graph_write_batch_size: int = 500) -> DrawingIngestionConfig:
    return DrawingIngestionConfig(enabled=True, graph_write_batch_size=graph_write_batch_size)


def _make_service(metadata, *, vector_store=None, graph_store=None, config=None):
    return DrawingIngestionService(
        config=config or _cfg(),
        embeddings=_FakeEmbeddings(),
        vector_store=vector_store or _RecordingVectorStore(),
        metadata_store=metadata,  # type: ignore[arg-type]
        embedding_model_name="test-embed",
        graph_store=graph_store,
    )


async def _seed_linked(metadata, pages, pairings=None, residue=None):
    src = Source(
        source_id="src-1",
        status="linked",
        file_hash="abc",
        knowledge_id="k1",
        metadata={
            "source_format": "pdf",
            "file_name": "simple.pdf",
            "drawing_linking": {
                "deterministic_pairings": [p.to_dict() for p in (pairings or [])],
                "fuzzy_merges": [],
                "llm_residue": residue or [],
            },
        },
    )
    await metadata.create_source(src)
    rows = [{"page_number": p.page_number, "data": {"analysis": p.to_dict()}} for p in pages]
    await metadata.upsert_page_analyses(src.source_id, rows)
    return src


def _dc(cid: str, klass: str = "resistor") -> DetectedComponent:
    return DetectedComponent(
        component_id=cid,
        symbol_class=klass,
        label=cid,
        bbox=[0, 0, 10, 10],
        ports=[],
        properties=None,
    )


def _dp(page: int, components) -> DrawingPageAnalysis:
    return DrawingPageAnalysis(
        page_number=page,
        components=components,
        connections=[],
        off_page_connectors=[],
        domain="electrical",
        page_type="drawing",
        notes=[],
    )


async def test_ingest_requires_linked_status() -> None:
    metadata = _InMemoryMetadataStore()
    svc = _make_service(metadata)
    bad = Source(source_id="s", status="extracted", file_hash="h", knowledge_id="k1", metadata={})
    await metadata.create_source(bad)
    with pytest.raises(IngestionError, match="(?i)requires status"):
        await svc.ingest("s")


async def test_ingest_missing_source_raises() -> None:
    metadata = _InMemoryMetadataStore()
    svc = _make_service(metadata)
    with pytest.raises(IngestionError, match="(?i)source not found"):
        await svc.ingest("nope")


async def test_ingest_emits_one_vector_per_component() -> None:
    p1 = _dp(1, [_dc("R1"), _dc("C1", "capacitor")])
    p2 = _dp(2, [_dc("R2")])
    metadata = _InMemoryMetadataStore()
    vstore = _RecordingVectorStore()
    svc = _make_service(metadata, vector_store=vstore)
    src = await _seed_linked(metadata, [p1, p2])
    src = await svc.ingest(src.source_id)
    assert src.status == "completed"
    assert len(vstore.upserted) == 3
    for p in vstore.upserted:
        assert p.payload["vector_role"] == "drawing_component"
        assert p.payload["source_id"] == "src-1"
        assert p.payload["source_type"] == "drawing"


async def test_ingest_writes_graph_entities_and_relations_when_store_present() -> None:
    p1 = _dp(1, [_dc("R1"), _dc("R2")])
    metadata = _InMemoryMetadataStore()
    gstore = _RecordingGraphStore()
    svc = _make_service(metadata, graph_store=gstore)
    src = await _seed_linked(metadata, [p1])
    await svc.ingest(src.source_id)
    assert len(gstore.entities_calls) == 1
    # No relations (no connections on page), but add_relations should NOT be called in that case
    # (or may be called once with an empty list — tolerate both)
    for _, rels in gstore.relations_calls:
        assert isinstance(rels, list)


async def test_ingest_batches_graph_relations() -> None:
    """graph_write_batch_size=2 with 5 same-page connections → 3 batched add_relations calls."""
    components = [_dc(f"R{i}") for i in range(6)]
    connections = [
        DetectedConnection(from_component=f"R{i}", to_component=f"R{i + 1}", wire_style="signal") for i in range(5)
    ]
    p1 = DrawingPageAnalysis(
        page_number=1,
        components=components,
        connections=connections,
        off_page_connectors=[],
        domain="electrical",
        page_type="drawing",
        notes=[],
    )
    metadata = _InMemoryMetadataStore()
    gstore = _RecordingGraphStore()
    svc = _make_service(metadata, graph_store=gstore, config=_cfg(graph_write_batch_size=2))
    src = await _seed_linked(metadata, [p1])
    await svc.ingest(src.source_id)
    # 5 relations / batch 2 = 3 calls (2, 2, 1)
    assert len(gstore.relations_calls) == 3
    assert sum(len(rels) for _, rels in gstore.relations_calls) == 5


async def test_ingest_idempotent_on_completed_source() -> None:
    p1 = _dp(1, [_dc("R1")])
    metadata = _InMemoryMetadataStore()
    vstore = _RecordingVectorStore()
    svc = _make_service(metadata, vector_store=vstore)
    src = await _seed_linked(metadata, [p1])
    src1 = await svc.ingest(src.source_id)
    upsert_count_after_first = len(vstore.upserted)
    # Second ingest → no additional upserts
    src2 = await svc.ingest(src1.source_id)
    assert src2.status == "completed"
    assert len(vstore.upserted) == upsert_count_after_first


async def test_ingest_without_graph_store_still_succeeds() -> None:
    """graph_store is optional — ingest must complete even without one."""
    p1 = _dp(1, [_dc("R1")])
    metadata = _InMemoryMetadataStore()
    svc = _make_service(metadata)  # no graph_store
    src = await _seed_linked(metadata, [p1])
    src = await svc.ingest(src.source_id)
    assert src.status == "completed"


async def test_ingest_sets_chunk_count_to_component_count() -> None:
    p1 = _dp(1, [_dc("R1"), _dc("R2"), _dc("R3")])
    metadata = _InMemoryMetadataStore()
    svc = _make_service(metadata)
    src = await _seed_linked(metadata, [p1])
    src = await svc.ingest(src.source_id)
    assert src.chunk_count == 3


async def test_ingest_passes_llm_residue_to_graph_mapper() -> None:
    p1 = _dp(1, [_dc("v1", "valve_ball")])
    p2 = _dp(2, [_dc("v2", "valve_ball")])
    residue = [
        {
            "page_a": 1,
            "component_a": "v1",
            "page_b": 2,
            "component_b": "v2",
            "confidence": 0.85,
            "rationale": "both labelled V-101",
        }
    ]
    metadata = _InMemoryMetadataStore()
    gstore = _RecordingGraphStore()
    svc = _make_service(metadata, graph_store=gstore)
    src = await _seed_linked(metadata, [p1, p2], residue=residue)
    await svc.ingest(src.source_id)
    # At least one MENTIONS relation from the LLM residue
    all_relations = [r for _, rels in gstore.relations_calls for r in rels]
    mentions = [r for r in all_relations if r.relation_type == "MENTIONS"]
    assert len(mentions) == 1
    assert mentions[0].from_entity == "v1"
    assert mentions[0].to_entity == "v2"
