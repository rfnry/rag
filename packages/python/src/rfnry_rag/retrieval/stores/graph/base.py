from typing import Protocol

from rfnry_rag.retrieval.stores.graph.models import GraphEntity, GraphRelation, GraphResult


class BaseGraphStore(Protocol):
    """Protocol for graph store implementations (Neo4j, etc.)."""

    async def initialize(self) -> None:
        """Create indexes and constraints. Idempotent."""
        ...

    async def add_entities(
        self,
        source_id: str,
        knowledge_id: str | None,
        entities: list[GraphEntity],
    ) -> None:
        """Upsert entities into the graph. Deduplicates by (name, entity_type, knowledge_id)."""
        ...

    async def add_relations(
        self,
        source_id: str,
        relations: list[GraphRelation],
    ) -> None:
        """Create relationships between existing entities."""
        ...

    async def query_graph(
        self,
        query: str,
        knowledge_id: str | None = None,
        entity_types: list[str] | None = None,
        max_hops: int = 2,
        top_k: int = 10,
    ) -> list[GraphResult]:
        """Find entities matching the query and traverse their connections."""
        ...

    async def delete_by_source(self, source_id: str) -> None:
        """Remove all entities, relationships, and pages introduced by this source."""
        ...

    async def shutdown(self) -> None:
        """Close driver connections."""
        ...
