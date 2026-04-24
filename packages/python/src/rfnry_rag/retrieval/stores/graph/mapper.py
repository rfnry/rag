"""Map structured analysis output (PageAnalysis, DocumentSynthesis) to graph entities and relations."""

from __future__ import annotations

import re

from rfnry_rag.common.logging import get_logger
from rfnry_rag.retrieval.modules.ingestion.analyze.models import DocumentSynthesis, PageAnalysis
from rfnry_rag.retrieval.stores.graph.models import GraphEntity, GraphRelation

logger = get_logger(__name__)

_ELECTRICAL_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bmotor\b", re.IGNORECASE), "motor"),
    (re.compile(r"\bbreaker\b|\bCB[-\s]", re.IGNORECASE), "breaker"),
    (re.compile(r"\bVFD\b|\bdrive\b", re.IGNORECASE), "vfd"),
    (re.compile(r"\bPLC\b", re.IGNORECASE), "plc"),
    (re.compile(r"\bpanel\b|\bMCC\b", re.IGNORECASE), "panel"),
]

_MECHANICAL_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bvalve\b", re.IGNORECASE), "valve"),
    (re.compile(r"\bpump\b", re.IGNORECASE), "pump"),
    (re.compile(r"\btank\b", re.IGNORECASE), "tank"),
]

_RELATIONSHIP_KEYWORDS: list[tuple[list[str], str]] = [
    (["power", "feed"], "POWERED_BY"),
    (["control"], "CONTROLLED_BY"),
    (["flow"], "FLOWS_TO"),
    (["connect"], "CONNECTS_TO"),
]


def _infer_entity_type(category: str, name: str) -> str:
    """Infer a graph-friendly entity type from the category and name."""
    patterns = _ELECTRICAL_PATTERNS + _MECHANICAL_PATTERNS
    for pattern, entity_type in patterns:
        if pattern.search(name):
            return entity_type

    if category:
        return category.lower()
    return "component"


def _classify_relationship(relationship: str) -> str | None:
    """Classify a cross-reference relationship string into a graph relationship type.

    Returns None when no keyword matches — the caller is responsible for
    dropping unclassifiable relations rather than defaulting to a fallback type.
    """
    lower = relationship.lower()
    for keywords, rel_type in _RELATIONSHIP_KEYWORDS:
        if any(kw in lower for kw in keywords):
            return rel_type
    return None


def page_entities_to_graph(page: PageAnalysis, source_id: str) -> list[GraphEntity]:
    """Convert PageAnalysis.entities to GraphEntity list."""
    return [
        GraphEntity(
            name=e.name,
            entity_type=_infer_entity_type(e.category, e.name),
            category=e.category,
            value=e.value,
            properties={
                "context": e.context,
                "page_number": page.page_number,
                "page_type": page.page_type,
                "source_id": source_id,
            },
        )
        for e in page.entities
    ]


def cross_refs_to_graph_relations(
    synthesis: DocumentSynthesis,
    page_analyses: list[PageAnalysis],
    knowledge_id: str | None,
) -> list[GraphRelation]:
    """Convert DocumentSynthesis.cross_references to GraphRelation list via shared entities."""
    entity_lookup: dict[str, str] = {}
    for pa in page_analyses:
        for e in pa.entities:
            entity_lookup[e.name] = _infer_entity_type(e.category, e.name)

    relations: list[GraphRelation] = []
    for xref in synthesis.cross_references:
        if len(xref.shared_entities) < 2:
            continue

        relation_type = _classify_relationship(xref.relationship)
        if relation_type is None:
            logger.debug(
                "dropping unclassifiable cross-reference relationship: %r",
                xref.relationship,
            )
            continue

        for i, e1_name in enumerate(xref.shared_entities):
            for e2_name in xref.shared_entities[i + 1 :]:
                if e1_name in entity_lookup and e2_name in entity_lookup:
                    relations.append(
                        GraphRelation(
                            from_entity=e1_name,
                            from_type=entity_lookup[e1_name],
                            to_entity=e2_name,
                            to_type=entity_lookup[e2_name],
                            relation_type=relation_type,
                            knowledge_id=knowledge_id,
                            context=xref.relationship,
                        )
                    )

    return relations
