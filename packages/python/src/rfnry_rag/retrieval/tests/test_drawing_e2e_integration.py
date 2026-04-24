"""End-to-end DrawingIngestion: real ezdxf, real mapper, stubbed stores, zero LLM."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from rfnry_rag.retrieval.common.models import VectorPoint
from rfnry_rag.retrieval.modules.ingestion.drawing.config import DrawingIngestionConfig
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

    async def find_by_hash(self, hash_value, knowledge_id):
        for s in self._sources.values():
            if s.file_hash == hash_value and s.knowledge_id == knowledge_id:
                return s
        return None

    async def upsert_page_analyses(self, source_id, analyses):
        existing = {r["page_number"]: r for r in self._pages.get(source_id, [])}
        for r in analyses:
            prior = existing.get(
                r["page_number"], {"page_number": r["page_number"], "data": {}}
            )
            merged = {**prior.get("data", {}), **r.get("data", {})}
            existing[r["page_number"]] = {
                "page_number": r["page_number"],
                "data": merged,
            }
        self._pages[source_id] = list(existing.values())

    async def get_page_analyses(self, source_id):
        return list(self._pages.get(source_id, []))

    async def get_source(self, source_id):
        return self._sources.get(source_id)


class _RecordingVectorStore:
    def __init__(self) -> None:
        self.init_size: int | None = None
        self.upserted: list[VectorPoint] = []

    async def initialize(self, vector_size):
        self.init_size = vector_size

    async def upsert(self, points):
        self.upserted.extend(points)


class _RecordingGraphStore:
    def __init__(self) -> None:
        self.entities: list = []
        self.relations: list = []
        self.entities_calls = 0
        self.relations_calls = 0

    async def add_entities(self, source_id, knowledge_id, entities):
        self.entities_calls += 1
        self.entities.extend(entities)

    async def add_relations(self, source_id, relations):
        self.relations_calls += 1
        self.relations.extend(relations)


class _FakeEmbeddings:
    async def embed(self, texts):
        # Fixed-size vectors so vector_store.initialize gets a consistent size
        return [[float(len(t)) / 10.0, 1.0, 0.0, 0.0] for t in texts]

    async def embedding_dimension(self) -> int:
        return 4


@pytest.fixture
def simple_rlc_dxf(tmp_path: Path) -> Path:
    """A 3-component schematic DXF: R1 - C1 - L1 wired in series.

    Blocks have width + height so bbox is non-degenerate; wires terminate
    inside each block's bbox (within the 2-unit tolerance).
    """
    import ezdxf
    path = tmp_path / "simple_rlc.dxf"
    doc = ezdxf.new()

    for name in ("resistor", "capacitor", "inductor"):
        blk = doc.blocks.new(name=name)
        blk.add_line((0, 0), (10, 0))
        blk.add_line((5, -3), (5, 3))

    msp = doc.modelspace()
    msp.add_blockref("resistor", (0, 0))
    msp.add_blockref("capacitor", (50, 0))
    msp.add_blockref("inductor", (100, 0))

    # Wires between adjacent components, terminating inside their bboxes.
    msp.add_line((5, 0), (55, 0))
    msp.add_line((55, 0), (105, 0))

    doc.saveas(path)
    return path


def _make_service(metadata, *, vector_store=None, graph_store=None, config=None):
    return DrawingIngestionService(
        config=config or DrawingIngestionConfig(enabled=True),
        embeddings=_FakeEmbeddings(),
        vector_store=vector_store or _RecordingVectorStore(),
        metadata_store=metadata,  # type: ignore[arg-type]
        embedding_model_name="test-embed",
        graph_store=graph_store,
    )


async def test_dxf_end_to_end_completes_all_phases(simple_rlc_dxf: Path) -> None:
    metadata = _InMemoryMetadataStore()
    vstore = _RecordingVectorStore()
    gstore = _RecordingGraphStore()
    svc = _make_service(metadata, vector_store=vstore, graph_store=gstore)

    src = await svc.render(str(simple_rlc_dxf), knowledge_id="k1")
    assert src.status == "rendered"

    src = await svc.extract(src.source_id)
    assert src.status == "extracted"

    src = await svc.link(src.source_id)
    assert src.status == "linked"

    src = await svc.ingest(src.source_id)
    assert src.status == "completed"


async def test_dxf_end_to_end_emits_three_components_and_two_connections(
    simple_rlc_dxf: Path,
) -> None:
    metadata = _InMemoryMetadataStore()
    vstore = _RecordingVectorStore()
    gstore = _RecordingGraphStore()
    svc = _make_service(metadata, vector_store=vstore, graph_store=gstore)

    src = await svc.render(str(simple_rlc_dxf), knowledge_id="k1")
    src = await svc.extract(src.source_id)
    src = await svc.link(src.source_id)
    src = await svc.ingest(src.source_id)

    # One embedding per component
    assert len(vstore.upserted) == 3
    classes = {p.payload["symbol_class"] for p in vstore.upserted}
    assert classes == {"resistor", "capacitor", "inductor"}

    # Graph writes
    assert gstore.entities_calls == 1
    assert len(gstore.entities) == 3
    # 2 same-page connections (R1-C1, C1-L1)
    assert any(
        r.from_entity and r.to_entity for r in gstore.relations
    ), "expected at least one relation"


async def test_dxf_end_to_end_consumer_symbol_library_override(tmp_path: Path) -> None:
    """Consumer can replace the symbol library with in-house vocabulary."""
    import ezdxf
    path = tmp_path / "custom.dxf"
    doc = ezdxf.new()
    blk = doc.blocks.new(name="widget_alpha")
    blk.add_line((0, 0), (10, 0))
    blk.add_line((5, -3), (5, 3))
    msp = doc.modelspace()
    msp.add_blockref("widget_alpha", (0, 0))
    doc.saveas(path)

    cfg = DrawingIngestionConfig(
        enabled=True,
        symbol_library={"custom": ["widget_alpha", "widget_beta"]},
    )
    metadata = _InMemoryMetadataStore()
    svc = _make_service(metadata, config=cfg)

    src = await svc.render(str(path), knowledge_id="k1")
    src = await svc.extract(src.source_id)
    rows = await metadata.get_page_analyses(src.source_id)
    analysis = rows[0]["data"]["analysis"]
    assert len(analysis["components"]) == 1
    c = analysis["components"][0]
    assert c["symbol_class"] == "widget_alpha"
    assert c["properties"]["domain"] == "custom"


async def test_dxf_end_to_end_file_hash_idempotent_on_second_run(
    simple_rlc_dxf: Path,
) -> None:
    metadata = _InMemoryMetadataStore()
    svc = _make_service(metadata)
    src_a = await svc.render(str(simple_rlc_dxf), knowledge_id="k1")
    src_b = await svc.render(str(simple_rlc_dxf), knowledge_id="k1")
    assert src_a.source_id == src_b.source_id
    assert src_a.file_hash == src_b.file_hash
