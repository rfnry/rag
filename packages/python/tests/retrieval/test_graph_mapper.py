"""Graph mapper: page_entities_to_graph + cross_refs_to_graph_relations, config-driven."""

from __future__ import annotations

from rfnry_knowledge.config.entity import EntityIngestionConfig
from rfnry_knowledge.ingestion.structured.models import (
    CrossReference,
    DiscoveredEntity,
    DocumentSynthesis,
    PageAnalysis,
)
from rfnry_knowledge.stores.graph.mapper import (
    cross_refs_to_graph_relations,
    page_entities_to_graph,
)

_ELECTRICAL_CONFIG = EntityIngestionConfig(
    entity_type_patterns=[
        (r"\bmotor\b", "motor"),
        (r"\bbreaker\b|\bCB[-\s]", "breaker"),
        (r"\bVFD\b|\bdrive\b", "vfd"),
        (r"\bPLC\b", "plc"),
        (r"\bpanel\b|\bMCC\b", "panel"),
        (r"\bvalve\b", "valve"),
        (r"\bpump\b", "pump"),
        (r"\btank\b", "tank"),
    ],
    relationship_keyword_map={
        "power": "POWERED_BY",
        "feed": "POWERED_BY",
        "control": "CONTROLLED_BY",
        "flow": "FLOWS_TO",
        "connect": "CONNECTS_TO",
    },
    unclassified_relation_default=None,  # preserve the old "drop on miss" semantics
)


# ---- page_entities_to_graph ----


def test_page_entities_to_graph_with_electrical_config() -> None:
    page = PageAnalysis(
        page_number=1,
        description="Electrical schematic",
        entities=[
            DiscoveredEntity(name="Motor M1", category="electrical_component", context="main motor", value="480V"),
            DiscoveredEntity(name="Breaker CB-3", category="electrical_component", context="feeder breaker"),
        ],
        page_type="electrical_schematic",
    )
    result = page_entities_to_graph(page, source_id="src-1", config=_ELECTRICAL_CONFIG)
    assert len(result) == 2
    assert result[0].entity_type == "motor"
    assert result[1].entity_type == "breaker"


def test_page_entities_to_graph_default_config_uses_category() -> None:
    """Empty patterns -> type falls through to category.lower()."""
    page = PageAnalysis(
        page_number=1,
        description="Agnostic doc",
        entities=[
            DiscoveredEntity(name="Plaintiff", category="LegalParty", context="party A"),
            DiscoveredEntity(name="Contract-001", category="Contract", context=""),
        ],
    )
    result = page_entities_to_graph(page, source_id="src-1", config=EntityIngestionConfig())
    assert result[0].entity_type == "legalparty"
    assert result[1].entity_type == "contract"


def test_page_entities_to_graph_empty_category_falls_back_to_entity() -> None:
    page = PageAnalysis(
        page_number=1,
        description="",
        entities=[DiscoveredEntity(name="X", category="", context="")],
    )
    result = page_entities_to_graph(page, source_id="s", config=EntityIngestionConfig())
    assert result[0].entity_type == "entity"


def test_page_entities_empty() -> None:
    page = PageAnalysis(page_number=1, description="", entities=[])
    assert page_entities_to_graph(page, source_id="s", config=EntityIngestionConfig()) == []


# ---- cross_refs_to_graph_relations ----


def _two_entity_pages() -> list[PageAnalysis]:
    return [
        PageAnalysis(
            page_number=1,
            description="P1",
            entities=[
                DiscoveredEntity(name="Motor M1", category="electrical_component", context=""),
                DiscoveredEntity(name="Breaker CB-3", category="electrical_component", context=""),
            ],
        ),
    ]


def test_cross_refs_classifies_via_keyword_map() -> None:
    synthesis = DocumentSynthesis(
        cross_references=[
            CrossReference(
                source_page=1,
                target_page=2,
                relationship="power feed from breaker to motor",
                shared_entities=["Motor M1", "Breaker CB-3"],
            ),
        ]
    )
    relations = cross_refs_to_graph_relations(
        synthesis,
        _two_entity_pages(),
        knowledge_id="k",
        config=_ELECTRICAL_CONFIG,
    )
    assert len(relations) == 1
    assert relations[0].relation_type == "POWERED_BY"
    assert relations[0].from_entity in {"Motor M1", "Breaker CB-3"}


def test_cross_refs_unclassifiable_uses_default_mentions_when_set() -> None:
    """New Phase-D behavior: unknown relationship becomes MENTIONS edge."""
    cfg = EntityIngestionConfig()  # default unclassified_relation_default="MENTIONS"
    synthesis = DocumentSynthesis(
        cross_references=[
            CrossReference(
                source_page=1,
                target_page=2,
                relationship="foobarbaz-no-keywords",
                shared_entities=["Motor M1", "Breaker CB-3"],
            ),
        ]
    )
    relations = cross_refs_to_graph_relations(
        synthesis,
        _two_entity_pages(),
        knowledge_id="k",
        config=cfg,
    )
    assert len(relations) == 1
    assert relations[0].relation_type == "MENTIONS"


def test_cross_refs_unclassifiable_dropped_when_default_is_none() -> None:
    """Opt-in strict mode: consumer sets unclassified_relation_default=None."""
    cfg = EntityIngestionConfig(unclassified_relation_default=None)
    synthesis = DocumentSynthesis(
        cross_references=[
            CrossReference(
                source_page=1,
                target_page=2,
                relationship="foobarbaz-no-keywords",
                shared_entities=["Motor M1", "Breaker CB-3"],
            ),
        ]
    )
    relations = cross_refs_to_graph_relations(
        synthesis,
        _two_entity_pages(),
        knowledge_id="k",
        config=cfg,
    )
    assert relations == []


def test_cross_refs_skips_single_entity() -> None:
    synthesis = DocumentSynthesis(
        cross_references=[
            CrossReference(source_page=1, target_page=2, relationship="x", shared_entities=["Motor M1"]),
        ]
    )
    relations = cross_refs_to_graph_relations(
        synthesis,
        _two_entity_pages(),
        knowledge_id="k",
        config=_ELECTRICAL_CONFIG,
    )
    assert relations == []


def test_cross_refs_skips_unknown_entities() -> None:
    synthesis = DocumentSynthesis(
        cross_references=[
            CrossReference(
                source_page=1, target_page=2, relationship="connection", shared_entities=["Motor M1", "NotInAnyPage"]
            ),
        ]
    )
    relations = cross_refs_to_graph_relations(
        synthesis,
        _two_entity_pages(),
        knowledge_id="k",
        config=_ELECTRICAL_CONFIG,
    )
    assert relations == []


def test_cross_refs_pairwise_with_three_entities() -> None:
    pages = [
        PageAnalysis(
            page_number=1,
            description="P1",
            entities=[
                DiscoveredEntity(name="Motor M1", category="electrical_component", context=""),
                DiscoveredEntity(name="Breaker CB-3", category="electrical_component", context=""),
                DiscoveredEntity(name="Panel MCC-1", category="electrical_component", context=""),
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
        ]
    )
    relations = cross_refs_to_graph_relations(
        synthesis,
        pages,
        knowledge_id="k",
        config=_ELECTRICAL_CONFIG,
    )
    assert len(relations) == 3
