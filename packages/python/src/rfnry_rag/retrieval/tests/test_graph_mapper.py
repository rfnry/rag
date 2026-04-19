from rfnry_rag.retrieval.modules.ingestion.analyze.models import (
    CrossReference,
    DiscoveredEntity,
    DocumentSynthesis,
    PageAnalysis,
)
from rfnry_rag.retrieval.stores.graph.models import GraphEntity, GraphRelation


def test_page_entities_to_graph_basic():
    from rfnry_rag.retrieval.stores.graph.mapper import page_entities_to_graph

    page = PageAnalysis(
        page_number=1,
        description="Electrical schematic",
        entities=[
            DiscoveredEntity(name="Motor M1", category="electrical_component", context="main motor", value="480V"),
            DiscoveredEntity(name="Breaker CB-3", category="electrical_component", context="feeder breaker"),
        ],
        page_type="electrical_schematic",
    )
    result = page_entities_to_graph(page, source_id="src-1")

    assert len(result) == 2
    assert all(isinstance(e, GraphEntity) for e in result)
    assert result[0].name == "Motor M1"
    assert result[0].entity_type == "motor"
    assert result[0].category == "electrical_component"
    assert result[0].value == "480V"
    assert result[0].properties["page_number"] == 1
    assert result[1].name == "Breaker CB-3"
    assert result[1].entity_type == "breaker"


def test_infer_entity_type_electrical():
    from rfnry_rag.retrieval.stores.graph.mapper import _infer_entity_type

    assert _infer_entity_type("electrical_component", "Motor M1") == "motor"
    assert _infer_entity_type("electrical_component", "Breaker CB-3") == "breaker"
    assert _infer_entity_type("electrical_component", "VFD-3") == "vfd"
    assert _infer_entity_type("electrical_component", "PLC-1") == "plc"
    assert _infer_entity_type("electrical_component", "Panel MCC-1") == "panel"
    assert _infer_entity_type("electrical_component", "MCC-2") == "panel"


def test_infer_entity_type_mechanical():
    from rfnry_rag.retrieval.stores.graph.mapper import _infer_entity_type

    assert _infer_entity_type("mechanical_component", "Valve V-101") == "valve"
    assert _infer_entity_type("mechanical_component", "Pump P-3") == "pump"
    assert _infer_entity_type("mechanical_component", "Tank T-1") == "tank"


def test_infer_entity_type_fallback():
    from rfnry_rag.retrieval.stores.graph.mapper import _infer_entity_type

    assert _infer_entity_type("electrical_component", "Unknown Thing") == "electrical_component"
    assert _infer_entity_type("control_device", "Some Device") == "control_device"
    assert _infer_entity_type("", "Whatever") == "component"


def test_cross_refs_to_graph_relations():
    from rfnry_rag.retrieval.stores.graph.mapper import cross_refs_to_graph_relations

    page_analyses = [
        PageAnalysis(
            page_number=1,
            description="Page 1",
            entities=[
                DiscoveredEntity(name="Motor M1", category="electrical_component", context="main motor"),
                DiscoveredEntity(name="Breaker CB-3", category="electrical_component", context="feeder"),
            ],
        ),
        PageAnalysis(
            page_number=2,
            description="Page 2",
            entities=[
                DiscoveredEntity(name="Panel MCC-1", category="electrical_component", context="panel"),
            ],
        ),
    ]
    synthesis = DocumentSynthesis(
        cross_references=[
            CrossReference(
                source_page=1,
                target_page=2,
                relationship="power feed from breaker to motor",
                shared_entities=["Motor M1", "Breaker CB-3"],
            ),
        ],
    )

    relations = cross_refs_to_graph_relations(synthesis, page_analyses, knowledge_id="kb-1")

    assert len(relations) == 1
    assert isinstance(relations[0], GraphRelation)
    assert relations[0].from_entity == "Motor M1"
    assert relations[0].to_entity == "Breaker CB-3"
    assert relations[0].relation_type == "POWERED_BY"
    assert relations[0].knowledge_id == "kb-1"


def test_cross_refs_skips_single_entity():
    from rfnry_rag.retrieval.stores.graph.mapper import cross_refs_to_graph_relations

    page_analyses = [
        PageAnalysis(
            page_number=1,
            description="Page 1",
            entities=[
                DiscoveredEntity(name="Motor M1", category="electrical_component", context="motor"),
            ],
        ),
    ]
    synthesis = DocumentSynthesis(
        cross_references=[
            CrossReference(
                source_page=1,
                target_page=2,
                relationship="something",
                shared_entities=["Motor M1"],
            ),
        ],
    )

    relations = cross_refs_to_graph_relations(synthesis, page_analyses, knowledge_id="kb-1")
    assert len(relations) == 0


def test_cross_refs_skips_unknown_entities():
    from rfnry_rag.retrieval.stores.graph.mapper import cross_refs_to_graph_relations

    page_analyses = [
        PageAnalysis(
            page_number=1,
            description="Page 1",
            entities=[
                DiscoveredEntity(name="Motor M1", category="electrical_component", context="motor"),
            ],
        ),
    ]
    synthesis = DocumentSynthesis(
        cross_references=[
            CrossReference(
                source_page=1,
                target_page=2,
                relationship="connection",
                shared_entities=["Motor M1", "Unknown Entity"],
            ),
        ],
    )

    relations = cross_refs_to_graph_relations(synthesis, page_analyses, knowledge_id="kb-1")
    assert len(relations) == 0


def test_classify_relationship_keywords():
    from rfnry_rag.retrieval.stores.graph.mapper import _classify_relationship

    assert _classify_relationship("power feed from breaker") == "POWERED_BY"
    assert _classify_relationship("main power supply") == "POWERED_BY"
    assert _classify_relationship("control signal to VFD") == "CONTROLLED_BY"
    assert _classify_relationship("water flow to tank") == "FLOWS_TO"
    assert _classify_relationship("connects to panel") == "CONNECTS_TO"
    assert _classify_relationship("some random relationship") == "CONNECTS_TO"


def test_page_entities_empty():
    from rfnry_rag.retrieval.stores.graph.mapper import page_entities_to_graph

    page = PageAnalysis(page_number=1, description="Empty page", entities=[])
    result = page_entities_to_graph(page, source_id="src-1")
    assert result == []


def test_cross_refs_pairwise_with_three_entities():
    """Three shared entities should produce 3 pairwise relations."""
    from rfnry_rag.retrieval.stores.graph.mapper import cross_refs_to_graph_relations

    page_analyses = [
        PageAnalysis(
            page_number=1,
            description="Page 1",
            entities=[
                DiscoveredEntity(name="Motor M1", category="electrical_component", context="motor"),
                DiscoveredEntity(name="Breaker CB-3", category="electrical_component", context="breaker"),
                DiscoveredEntity(name="Panel MCC-1", category="electrical_component", context="panel"),
            ],
        ),
    ]
    synthesis = DocumentSynthesis(
        cross_references=[
            CrossReference(
                source_page=1,
                target_page=2,
                relationship="power connection",
                shared_entities=["Motor M1", "Breaker CB-3", "Panel MCC-1"],
            ),
        ],
    )

    relations = cross_refs_to_graph_relations(synthesis, page_analyses, knowledge_id="kb-1")
    assert len(relations) == 3
