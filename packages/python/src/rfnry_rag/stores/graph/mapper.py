"""Map structured analysis output (PageAnalysis, DocumentSynthesis) to graph entities and relations.

Vocabulary is consumer-supplied via ``GraphIngestionConfig``; the mapper carries
no built-in domain assumptions. The ``unclassified_relation_default`` setting
determines whether a cross-reference with no matching keyword becomes a
generic ``MENTIONS`` edge (default) or is dropped (``None``).
"""

from __future__ import annotations

import re

from rfnry_rag.common.logging import get_logger
from rfnry_rag.ingestion.analyze.models import DocumentSynthesis, PageAnalysis
from rfnry_rag.ingestion.graph.config import GraphIngestionConfig
from rfnry_rag.stores.graph.models import GraphEntity, GraphRelation

logger = get_logger(__name__)


def _infer_entity_type(
    category: str,
    name: str,
    config: GraphIngestionConfig,
) -> str:
    """Infer an entity type. Consumer patterns first, then category, then 'entity'."""
    for pattern_str, type_name in config.entity_type_patterns:
        if re.search(pattern_str, name, re.IGNORECASE):
            return type_name
    if category:
        return category.lower()
    return "entity"


def _classify_relationship(
    relationship: str,
    config: GraphIngestionConfig,
) -> str | None:
    """Classify a cross-reference relationship via the consumer's keyword map.

    Returns ``None`` when no keyword matches AND
    ``config.unclassified_relation_default is None`` (strict-drop mode).
    Otherwise falls back to ``unclassified_relation_default`` (default
    ``"MENTIONS"``).
    """
    lower = relationship.lower()
    for keyword, rel_type in config.relationship_keyword_map.items():
        if keyword.lower() in lower:
            return rel_type
    return config.unclassified_relation_default


def page_entities_to_graph(
    page: PageAnalysis,
    source_id: str,
    config: GraphIngestionConfig,
) -> list[GraphEntity]:
    """Convert PageAnalysis.entities to GraphEntity list using config-driven type inference."""
    return [
        GraphEntity(
            name=e.name,
            entity_type=_infer_entity_type(e.category, e.name, config),
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
    config: GraphIngestionConfig,
) -> list[GraphRelation]:
    """Convert DocumentSynthesis.cross_references to GraphRelation list."""
    entity_lookup: dict[str, str] = {}
    for pa in page_analyses:
        for e in pa.entities:
            entity_lookup[e.name] = _infer_entity_type(e.category, e.name, config)

    relations: list[GraphRelation] = []
    for xref in synthesis.cross_references:
        if len(xref.shared_entities) < 2:
            continue

        relation_type = _classify_relationship(xref.relationship, config)
        if relation_type is None:
            logger.debug(
                "dropping unclassifiable cross-reference (no keyword match, no fallback): %r",
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
