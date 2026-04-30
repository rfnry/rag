"""Drawing graph mapper: no shared_entities>=2 filter, threads bbox/ports, uses relation_vocabulary."""

from __future__ import annotations

from rfnry_rag.ingestion.drawing.config import DrawingIngestionConfig
from rfnry_rag.ingestion.drawing.models import (
    DetectedComponent,
    DetectedConnection,
    DrawingPageAnalysis,
    Port,
)
from rfnry_rag.stores.graph.drawing_mapper import (
    component_to_graph_entity,
    connection_to_graph_relation,
    drawing_to_graph,
)


def _cfg() -> DrawingIngestionConfig:
    return DrawingIngestionConfig(enabled=True)


def _component(cid: str = "R1", klass: str = "resistor", **kwargs) -> DetectedComponent:
    return DetectedComponent(
        component_id=cid,
        symbol_class=klass,
        label=kwargs.get("label"),
        bbox=kwargs.get("bbox", [10, 20, 30, 40]),
        ports=kwargs.get("ports", []),
        properties=kwargs.get("properties"),
    )


def _page(components, connections=None, off_page=None, page=1) -> DrawingPageAnalysis:
    return DrawingPageAnalysis(
        page_number=page,
        components=components,
        connections=connections or [],
        off_page_connectors=off_page or [],
        domain="electrical",
        page_type="drawing",
        notes=[],
    )


# ---- component_to_graph_entity ----


def test_component_maps_with_bbox_and_ports_in_properties() -> None:
    c = _component(
        cid="R1",
        klass="resistor",
        label="R1 10k",
        bbox=[10, 20, 30, 40],
        ports=[Port(port_id="a", position=[10, 30]), Port(port_id="b", position=[40, 30])],
        properties={"tolerance": "5%"},
    )
    entity = component_to_graph_entity(c, source_id="src-1", page_number=1, domain="electrical")
    assert entity.name == "R1"
    assert entity.entity_type == "resistor"
    assert entity.properties["bbox"] == [10, 20, 30, 40]
    assert len(entity.properties["ports"]) == 2
    assert entity.properties["ports"][0]["port_id"] == "a"
    assert entity.properties["page_number"] == 1
    assert entity.properties["domain"] == "electrical"
    assert entity.properties["source_id"] == "src-1"
    assert entity.properties["label"] == "R1 10k"
    # Consumer-supplied component.properties flow through too (namespaced so we don't collide)
    assert entity.properties.get("tolerance") == "5%"


def test_component_without_optional_fields_still_maps() -> None:
    c = _component(cid="C1", klass="capacitor", label=None, ports=[], properties=None)
    entity = component_to_graph_entity(c, source_id="src-1", page_number=2, domain="electrical")
    assert entity.name == "C1"
    assert entity.properties["ports"] == []
    assert entity.properties["label"] == ""  # normalised to empty string, not None


# ---- connection_to_graph_relation ----


def test_connection_maps_to_allowlisted_relation_type_via_vocabulary() -> None:
    cfg = _cfg()
    conn = DetectedConnection(from_component="R1", to_component="C2", wire_style="pneumatic")
    type_lookup = {"R1": "resistor", "C2": "capacitor"}
    rel = connection_to_graph_relation(
        conn,
        source_id="s1",
        page_number=1,
        config=cfg,
        component_type_lookup=type_lookup,
        knowledge_id="k1",
    )
    assert rel.relation_type == "FLOWS_TO"
    assert rel.knowledge_id == "k1"
    assert rel.from_entity == "R1"
    assert rel.to_entity == "C2"
    assert rel.from_type == "resistor"
    assert rel.to_type == "capacitor"


def test_unknown_wire_style_falls_back_to_connects_to() -> None:
    cfg = _cfg()
    conn = DetectedConnection(from_component="R1", to_component="C2", wire_style="bogus_custom")
    rel = connection_to_graph_relation(
        conn,
        source_id="s1",
        page_number=1,
        config=cfg,
        component_type_lookup={},
    )
    assert rel.relation_type == "CONNECTS_TO"


def test_none_wire_style_falls_back_to_connects_to() -> None:
    cfg = _cfg()
    conn = DetectedConnection(from_component="R1", to_component="C2", wire_style=None)
    rel = connection_to_graph_relation(
        conn,
        source_id="s1",
        page_number=1,
        config=cfg,
        component_type_lookup={},
    )
    assert rel.relation_type == "CONNECTS_TO"


def test_connection_context_encodes_drawing_metadata() -> None:
    cfg = _cfg()
    conn = DetectedConnection(
        from_component="R1",
        to_component="C2",
        from_port="b",
        to_port="a",
        net_label="N5",
        wire_style="signal",
        properties={"cross_sheet": True, "from_page": 1, "to_page": 3},
    )
    rel = connection_to_graph_relation(
        conn,
        source_id="s1",
        page_number=1,
        config=cfg,
        component_type_lookup={},
    )
    # Context is a deterministic key=value;key=value string covering the drawing fields
    assert "wire_style=signal" in rel.context
    assert "from_port=b" in rel.context
    assert "to_port=a" in rel.context
    assert "net=N5" in rel.context
    assert "cross_sheet=true" in rel.context.lower()


def test_missing_component_types_default_to_component() -> None:
    cfg = _cfg()
    conn = DetectedConnection(from_component="UNKNOWN_A", to_component="UNKNOWN_B", wire_style="signal")
    rel = connection_to_graph_relation(
        conn,
        source_id="s1",
        page_number=1,
        config=cfg,
        component_type_lookup={},
    )
    assert rel.from_type == "component"
    assert rel.to_type == "component"


# ---- drawing_to_graph (top-level orchestration) ----


def test_drawing_to_graph_single_component_crossref_produces_relation_not_skipped() -> None:
    """The existing analyze-path drops cross-refs with <2 shared entities.
    The drawing mapper MUST NOT apply that filter."""
    from rfnry_rag.ingestion.drawing.models import DetectedConnection as DC

    p1 = _page([_component("R1")], page=1)
    p2 = _page([_component("C1", "capacitor")], page=2)
    cfg = _cfg()
    cross_pair = DC(
        from_component="R1",
        to_component="C1",
        wire_style="signal",
        properties={"cross_sheet": True, "from_page": 1, "to_page": 2},
    )
    entities, relations = drawing_to_graph(
        pages=[p1, p2],
        deterministic_pairings=[cross_pair],
        source_id="src-1",
        config=cfg,
        knowledge_id="k1",
    )
    assert len(entities) == 2
    assert any(r.from_entity == "R1" and r.to_entity == "C1" for r in relations)
