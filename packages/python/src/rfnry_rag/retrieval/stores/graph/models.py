from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GraphEntity:
    """An entity node in the knowledge graph."""

    name: str
    entity_type: str
    category: str = ""
    value: str | None = None
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphRelation:
    """A relationship between two entities in the knowledge graph."""

    from_entity: str
    from_type: str
    to_entity: str
    to_type: str
    relation_type: str
    knowledge_id: str | None = None
    context: str = ""
    confidence: float = 1.0


@dataclass
class GraphPath:
    """A traversal path through the knowledge graph."""

    entities: list[str] = field(default_factory=list)
    relationships: list[str] = field(default_factory=list)
    description: str = ""


@dataclass
class GraphResult:
    """A graph traversal result: a seed entity with its connected subgraph."""

    entity: GraphEntity
    connected_entities: list[GraphEntity] = field(default_factory=list)
    paths: list[GraphPath] = field(default_factory=list)
    relevance_score: float = 0.0
