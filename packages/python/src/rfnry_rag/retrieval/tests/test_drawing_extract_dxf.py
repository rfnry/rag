"""Drawing extract (DXF): direct entity parse — zero LLM calls."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from rfnry_rag.retrieval.modules.ingestion.drawing.config import DrawingIngestionConfig
from rfnry_rag.retrieval.modules.ingestion.drawing.service import DrawingIngestionService


class _InMemoryMetadataStore:
    def __init__(self) -> None:
        self._sources: dict[str, Any] = {}
        self._pages: dict[str, list[dict]] = {}

    async def create_source(self, source) -> None:
        self._sources[source.source_id] = source

    async def update_source(self, source_id: str, **fields) -> None:
        src = self._sources[source_id]
        for k, v in fields.items():
            setattr(src, k, v)

    async def find_by_hash(self, hash_value: str, knowledge_id: str | None):
        for s in self._sources.values():
            if s.file_hash == hash_value and s.knowledge_id == knowledge_id:
                return s
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

    async def get_source(self, source_id: str):
        return self._sources.get(source_id)


def _make_service(metadata) -> DrawingIngestionService:
    # DXF path does NOT require lm_client (zero LLM calls)
    cfg = DrawingIngestionConfig(enabled=True)
    return DrawingIngestionService(
        config=cfg,
        embeddings=SimpleNamespace(),
        vector_store=SimpleNamespace(),
        metadata_store=metadata,  # type: ignore[arg-type]
        embedding_model_name="test-embed",
    )


@pytest.fixture
def dxf_with_two_resistors(tmp_path: Path) -> Path:
    """Create a DXF with two INSERT entities pointing to a 'resistor' block."""
    import ezdxf
    path = tmp_path / "two_resistors.dxf"
    doc = ezdxf.new()
    # Define a 'resistor' block
    blk = doc.blocks.new(name="resistor")
    blk.add_line((0, 0), (10, 0))
    blk.add_line((5, -3), (5, 3))
    msp = doc.modelspace()
    msp.add_blockref("resistor", (0, 0))
    msp.add_blockref("resistor", (20, 0))
    doc.saveas(path)
    return path


@pytest.fixture
def dxf_with_wire_between_two_pins(tmp_path: Path) -> Path:
    """Create a DXF with two INSERTs and a LINE connecting them.

    Blocks have both horizontal + vertical geometry so _bbox_of_block_insert
    produces a non-degenerate bbox. The connecting wire terminates within
    a couple of modelspace units of each bbox (within _CONNECTION_TOL).
    """
    import ezdxf
    path = tmp_path / "wire.dxf"
    doc = ezdxf.new()
    blk = doc.blocks.new(name="resistor")
    blk.add_line((0, 0), (10, 0))
    blk.add_line((5, -5), (5, 5))  # vertical stroke so bbox has height
    msp = doc.modelspace()
    msp.add_blockref("resistor", (0, 0))
    msp.add_blockref("resistor", (50, 0))
    # Wire ends that sit just inside each bbox (within _CONNECTION_TOL of its edge)
    msp.add_line((5, 5), (55, 5))
    doc.saveas(path)
    return path


async def test_extract_dxf_makes_zero_llm_calls(dxf_with_two_resistors: Path) -> None:
    metadata = _InMemoryMetadataStore()
    svc = _make_service(metadata)
    src = await svc.render(str(dxf_with_two_resistors), knowledge_id="k1")
    mock_analyze = AsyncMock()
    with patch(
        "rfnry_rag.retrieval.modules.ingestion.drawing.extract_pdf.b.AnalyzeDrawingPage",
        mock_analyze,
    ):
        src = await svc.extract(src.source_id)
    assert src.status == "extracted"
    assert mock_analyze.call_count == 0


async def test_extract_dxf_maps_insert_entities_to_components(
    dxf_with_two_resistors: Path,
) -> None:
    metadata = _InMemoryMetadataStore()
    svc = _make_service(metadata)
    src = await svc.render(str(dxf_with_two_resistors), knowledge_id="k1")
    src = await svc.extract(src.source_id)
    rows = await metadata.get_page_analyses(src.source_id)
    analysis = rows[0]["data"]["analysis"]
    resistors = [c for c in analysis["components"] if c["symbol_class"] == "resistor"]
    assert len(resistors) == 2


async def test_extract_dxf_maps_line_entities_to_connections(
    dxf_with_wire_between_two_pins: Path,
) -> None:
    metadata = _InMemoryMetadataStore()
    svc = _make_service(metadata)
    src = await svc.render(str(dxf_with_wire_between_two_pins), knowledge_id="k1")
    src = await svc.extract(src.source_id)
    rows = await metadata.get_page_analyses(src.source_id)
    analysis = rows[0]["data"]["analysis"]
    assert len(analysis["connections"]) >= 1


async def test_extract_dxf_preserves_render_artifacts(dxf_with_two_resistors: Path) -> None:
    """Extract must not clobber the page_image_b64 written by render."""
    metadata = _InMemoryMetadataStore()
    svc = _make_service(metadata)
    src = await svc.render(str(dxf_with_two_resistors), knowledge_id="k1")
    src = await svc.extract(src.source_id)
    rows = await metadata.get_page_analyses(src.source_id)
    assert "page_image_b64" in rows[0]["data"]
    assert "analysis" in rows[0]["data"]
    assert rows[0]["data"]["source_format"] == "dxf"


async def test_extract_dxf_idempotent_on_reentry(dxf_with_two_resistors: Path) -> None:
    metadata = _InMemoryMetadataStore()
    svc = _make_service(metadata)
    src = await svc.render(str(dxf_with_two_resistors), knowledge_id="k1")
    src1 = await svc.extract(src.source_id)
    src2 = await svc.extract(src1.source_id)
    assert src2.status == "extracted"


async def test_extract_dxf_classifies_by_symbol_library(tmp_path: Path) -> None:
    """A block named 'valve_gate' should classify as 'p_and_id' domain per default library."""
    import ezdxf
    path = tmp_path / "valve.dxf"
    doc = ezdxf.new()
    b = doc.blocks.new(name="valve_gate")
    b.add_line((0, 0), (5, 0))
    b.add_line((2, -2), (2, 2))
    msp = doc.modelspace()
    msp.add_blockref("valve_gate", (0, 0))
    doc.saveas(path)

    metadata = _InMemoryMetadataStore()
    svc = _make_service(metadata)
    src = await svc.render(str(path), knowledge_id="k1")
    src = await svc.extract(src.source_id)
    rows = await metadata.get_page_analyses(src.source_id)
    analysis = rows[0]["data"]["analysis"]
    assert len(analysis["components"]) == 1
    c = analysis["components"][0]
    assert c["symbol_class"] == "valve_gate"
    # properties.domain reflects the lookup
    assert c["properties"] is not None
    assert c["properties"].get("domain") == "p_and_id"


async def test_extract_dxf_emits_off_page_connectors_from_text(tmp_path: Path) -> None:
    """A TEXT entity reading '/A2' inside an INSERT bbox is emitted as bound OffPageConnector."""
    import ezdxf
    path = tmp_path / "opc_text.dxf"
    doc = ezdxf.new()
    blk = doc.blocks.new(name="resistor")
    blk.add_line((0, 0), (10, 0))
    blk.add_line((5, -3), (5, 3))
    msp = doc.modelspace()
    msp.add_blockref("resistor", (0, 0))
    text = msp.add_text("/A2")
    text.set_placement((5, 0))
    doc.saveas(path)

    metadata = _InMemoryMetadataStore()
    svc = _make_service(metadata)
    src = await svc.render(str(path), knowledge_id="k1")
    src = await svc.extract(src.source_id)
    rows = await metadata.get_page_analyses(src.source_id)
    analysis = rows[0]["data"]["analysis"]
    opcs = analysis["off_page_connectors"]
    assert len(opcs) == 1
    opc = opcs[0]
    assert opc["tag"] == "/A2"
    assert opc["target_hint"] == "/A2"
    component_ids = {c["component_id"] for c in analysis["components"]}
    assert opc["bound_component"] in component_ids


async def test_extract_dxf_emits_off_page_connectors_from_mtext(tmp_path: Path) -> None:
    """An MTEXT carrying formatting codes is stripped to plain text before regex match."""
    import ezdxf
    path = tmp_path / "opc_mtext.dxf"
    doc = ezdxf.new()
    blk = doc.blocks.new(name="resistor")
    blk.add_line((0, 0), (10, 0))
    blk.add_line((5, -3), (5, 3))
    msp = doc.modelspace()
    msp.add_blockref("resistor", (0, 0))
    msp.add_mtext("{\\C1;/A2}", dxfattribs={"insert": (5, 0)})
    doc.saveas(path)

    metadata = _InMemoryMetadataStore()
    svc = _make_service(metadata)
    src = await svc.render(str(path), knowledge_id="k1")
    src = await svc.extract(src.source_id)
    rows = await metadata.get_page_analyses(src.source_id)
    analysis = rows[0]["data"]["analysis"]
    opcs = analysis["off_page_connectors"]
    assert len(opcs) == 1
    opc = opcs[0]
    assert opc["tag"] == "/A2"
    # Formatting codes must not leak through
    assert "\\C1" not in opc["tag"]
    assert "{" not in opc["tag"]


async def test_extract_dxf_unbound_off_page_connector(tmp_path: Path) -> None:
    """A TEXT outside any component bbox emits an OffPageConnector with bound_component=None."""
    import ezdxf
    path = tmp_path / "opc_unbound.dxf"
    doc = ezdxf.new()
    blk = doc.blocks.new(name="resistor")
    blk.add_line((0, 0), (10, 0))
    blk.add_line((5, -3), (5, 3))
    msp = doc.modelspace()
    msp.add_blockref("resistor", (0, 0))
    # Place far away from the block bbox (well beyond _CONNECTION_TOL).
    text = msp.add_text("/A2")
    text.set_placement((500, 500))
    doc.saveas(path)

    metadata = _InMemoryMetadataStore()
    svc = _make_service(metadata)
    src = await svc.render(str(path), knowledge_id="k1")
    src = await svc.extract(src.source_id)
    rows = await metadata.get_page_analyses(src.source_id)
    analysis = rows[0]["data"]["analysis"]
    opcs = analysis["off_page_connectors"]
    assert len(opcs) == 1
    assert opcs[0]["tag"] == "/A2"
    assert opcs[0]["bound_component"] is None


async def test_extract_dxf_ignores_non_matching_text(tmp_path: Path) -> None:
    """A TEXT carrying a label (not an off-page tag) must NOT produce a connector."""
    import ezdxf
    path = tmp_path / "label_only.dxf"
    doc = ezdxf.new()
    blk = doc.blocks.new(name="resistor")
    blk.add_line((0, 0), (10, 0))
    blk.add_line((5, -3), (5, 3))
    msp = doc.modelspace()
    msp.add_blockref("resistor", (0, 0))
    text = msp.add_text("R1 10k")
    text.set_placement((5, 0))
    doc.saveas(path)

    metadata = _InMemoryMetadataStore()
    svc = _make_service(metadata)
    src = await svc.render(str(path), knowledge_id="k1")
    src = await svc.extract(src.source_id)
    rows = await metadata.get_page_analyses(src.source_id)
    analysis = rows[0]["data"]["analysis"]
    assert analysis["off_page_connectors"] == []


async def test_extract_dxf_per_layout_components(tmp_path: Path) -> None:
    """Modelspace + Layout1 each carry one INSERT → two DrawingPageAnalysis objects."""
    import ezdxf

    path = tmp_path / "two_layouts.dxf"
    doc = ezdxf.new()
    blk = doc.blocks.new(name="comp_a")
    blk.add_line((0, 0), (10, 0))
    blk.add_line((5, -3), (5, 3))

    msp = doc.modelspace()
    msp.add_blockref("comp_a", (0, 0))
    ps1 = doc.layouts.get("Layout1")
    ps1.add_blockref("comp_a", (50, 0))
    doc.saveas(path)

    metadata = _InMemoryMetadataStore()
    svc = _make_service(metadata)
    src = await svc.render(str(path), knowledge_id="k1")
    src = await svc.extract(src.source_id)

    rows = await metadata.get_page_analyses(src.source_id)
    rows.sort(key=lambda r: r["page_number"])
    assert [r["page_number"] for r in rows] == [1, 2]
    msp_components = rows[0]["data"]["analysis"]["components"]
    ps1_components = rows[1]["data"]["analysis"]["components"]
    assert len(msp_components) == 1
    assert len(ps1_components) == 1
    assert rows[0]["data"]["analysis"]["page_number"] == 1
    assert rows[1]["data"]["analysis"]["page_number"] == 2


async def test_extract_dxf_corrupt_mtext_does_not_fail(
    tmp_path: Path, monkeypatch
) -> None:
    """If MText.plain_text() raises, parse continues and the entity is skipped."""
    import ezdxf
    from ezdxf.entities import MText

    path = tmp_path / "opc_corrupt_mtext.dxf"
    doc = ezdxf.new()
    blk = doc.blocks.new(name="resistor")
    blk.add_line((0, 0), (10, 0))
    blk.add_line((5, -3), (5, 3))
    msp = doc.modelspace()
    msp.add_blockref("resistor", (0, 0))
    # One healthy TEXT alongside the broken MTEXT — proves the loop continues.
    text = msp.add_text("/A2")
    text.set_placement((5, 0))
    msp.add_mtext("/B7", dxfattribs={"insert": (5, 0)})
    doc.saveas(path)

    metadata = _InMemoryMetadataStore()
    svc = _make_service(metadata)
    # Render first; matplotlib's DXF frontend also calls plain_text(), so the
    # monkeypatch is installed only around the extract step under test.
    src = await svc.render(str(path), knowledge_id="k1")

    def _broken_plain_text(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("corrupt MTEXT")

    monkeypatch.setattr(MText, "plain_text", _broken_plain_text)
    src = await svc.extract(src.source_id)
    rows = await metadata.get_page_analyses(src.source_id)
    analysis = rows[0]["data"]["analysis"]
    tags = sorted(o["tag"] for o in analysis["off_page_connectors"])
    assert tags == ["/A2"]
