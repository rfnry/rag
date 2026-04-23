from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

from rfnry_rag.retrieval.common.logging import get_logger
from rfnry_rag.retrieval.stores.graph.models import GraphEntity, GraphPath, GraphRelation, GraphResult

logger = get_logger(__name__)

ALLOWED_RELATION_TYPES = frozenset(
    {
        "CONNECTS_TO",
        "POWERED_BY",
        "CONTROLLED_BY",
        "FEEDS",
        "FLOWS_TO",
        "REFERENCES",
        "MENTIONS",
    }
)

_INDEX_QUERIES = [
    "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE",
    "CREATE CONSTRAINT document_source_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.source_id IS UNIQUE",
    "CREATE CONSTRAINT page_id_unique IF NOT EXISTS FOR (p:Page) REQUIRE p.page_id IS UNIQUE",
    "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
    "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.entity_type)",
    "CREATE INDEX entity_knowledge IF NOT EXISTS FOR (e:Entity) ON (e.knowledge_id)",
]

_FULLTEXT_INDEX_QUERY = "CREATE FULLTEXT INDEX entity_search IF NOT EXISTS FOR (e:Entity) ON EACH [e.name, e.value]"

_ENTITY_MERGE_QUERY = """
MERGE (e:Entity {entity_id: $entity_id})
ON CREATE SET
    e.name = $name,
    e.entity_type = $entity_type,
    e.category = $category,
    e.value = $value,
    e.knowledge_id = $knowledge_id,
    e.source_ids = [$source_id],
    e.properties = $properties
ON MATCH SET
    e.value = COALESCE($value, e.value),
    e.properties = $properties,
    e.source_ids = CASE
        WHEN NOT $source_id IN e.source_ids
        THEN e.source_ids + $source_id
        ELSE e.source_ids
    END
"""

_SEED_QUERY = """
CALL db.index.fulltext.queryNodes('entity_search', $query_text)
YIELD node, score
WHERE ($knowledge_id IS NULL OR node.knowledge_id = $knowledge_id)
RETURN node, score
ORDER BY score DESC
LIMIT $seed_limit
"""

_SEED_QUERY_WITH_TYPES = """
CALL db.index.fulltext.queryNodes('entity_search', $query_text)
YIELD node, score
WHERE ($knowledge_id IS NULL OR node.knowledge_id = $knowledge_id)
  AND node.entity_type IN $entity_types
RETURN node, score
ORDER BY score DESC
LIMIT $seed_limit
"""

_TRAVERSE_QUERY = """
MATCH (seed:Entity {entity_id: $seed_id})
OPTIONAL MATCH path = (seed)-[r:CONNECTS_TO|POWERED_BY|CONTROLLED_BY|FEEDS|FLOWS_TO*1..{max_hops}]-(connected:Entity)
WHERE connected.knowledge_id = seed.knowledge_id OR connected.knowledge_id IS NULL
RETURN seed,
       connected,
       [rel IN relationships(path) | type(rel)] AS rel_types,
       [n IN nodes(path) | n.name] AS node_names,
       length(path) AS hops
ORDER BY hops ASC
"""

_DELETE_RELATIONS_QUERY = """
MATCH ()-[r]->()
WHERE r.source_id = $source_id
DELETE r
"""

_DELETE_SOURCE_FROM_ENTITIES_QUERY = """
MATCH (e:Entity)
WHERE $source_id IN e.source_ids
SET e.source_ids = [sid IN e.source_ids WHERE sid <> $source_id]
"""

_DELETE_ORPHANED_ENTITIES_QUERY = """
MATCH (e:Entity)
WHERE size(e.source_ids) = 0
DETACH DELETE e
"""

_DELETE_PAGES_QUERY = """
MATCH (p:Page {source_id: $source_id})
DETACH DELETE p
"""

_DELETE_DOCUMENT_QUERY = """
MATCH (d:Document {source_id: $source_id})
DETACH DELETE d
"""


def _normalize_name(name: str) -> str:
    """Case-fold and collapse whitespace for dedup."""
    return re.sub(r"\s+", " ", name.strip().lower())


def _compute_entity_id(name: str, entity_type: str, knowledge_id: str | None) -> str:
    """Deterministic entity ID from (name_normalized, entity_type, knowledge_id)."""
    normalized = _normalize_name(name)
    type_normalized = entity_type.strip().lower()
    kid = knowledge_id or ""
    key = f"{normalized}|{type_normalized}|{kid}"
    return hashlib.sha256(key.encode()).hexdigest()


def _validate_relation_type(rel_type: str) -> str:
    """Validate and normalize relationship type. Falls back to CONNECTS_TO."""
    normalized = rel_type.upper().replace(" ", "_")
    if normalized not in ALLOWED_RELATION_TYPES:
        return "CONNECTS_TO"
    return normalized


def _node_to_entity(node: dict[str, Any]) -> GraphEntity:
    """Convert a Neo4j node dict to a GraphEntity."""
    return GraphEntity(
        name=node.get("name", ""),
        entity_type=node.get("entity_type", ""),
        category=node.get("category", ""),
        value=node.get("value"),
        properties=node.get("properties", {}),
    )


try:
    from neo4j import AsyncGraphDatabase
except ImportError:
    AsyncGraphDatabase = None  # type: ignore[assignment, misc]


@dataclass
class Neo4jGraphStore:
    """Graph store backed by Neo4j with async driver."""

    uri: str
    username: str = field(default="neo4j", repr=False)
    password: str = field(default="", repr=False)
    database: str = "neo4j"
    query_timeout: float = 5.0
    # Single-process SDK defaults, not the server-scale defaults lifted from
    # Neo4j deployment guides (size 100, acquisition 60s). If the pool is
    # exhausted, a 5s wait is preferable to a 60s event-loop stall.
    max_connection_pool_size: int = 10
    connection_acquisition_timeout: float = 5.0
    connection_timeout: float = 5.0

    _driver: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        # The Neo4j community-edition default is "password" — rejecting empty
        # passwords forces operators to be explicit rather than shipping with
        # the universally-known default string.
        from rfnry_rag.retrieval.common.errors import ConfigurationError
        if not self.password:
            raise ConfigurationError("Neo4jGraphStore requires a non-empty password")

    async def initialize(self) -> None:
        """Create the driver, verify connectivity, and ensure indexes exist."""
        if AsyncGraphDatabase is None:
            raise ImportError("neo4j package is required for Neo4jGraphStore. Install it with: pip install neo4j>=5.0")

        self._driver = AsyncGraphDatabase.driver(
            self.uri,
            auth=(self.username, self.password),
            max_connection_pool_size=self.max_connection_pool_size,
            connection_acquisition_timeout=self.connection_acquisition_timeout,
            connection_timeout=self.connection_timeout,
        )
        await self._driver.verify_connectivity()

        async with self._driver.session(database=self.database) as session:
            for query in _INDEX_QUERIES:
                await session.run(query)
            await session.run(_FULLTEXT_INDEX_QUERY)

        logger.info("neo4j graph store initialized (uri=%s, database=%s)", self.uri, self.database)

    async def add_entities(
        self,
        source_id: str,
        knowledge_id: str | None,
        entities: list[GraphEntity],
    ) -> None:
        """Upsert entities via MERGE on deterministic entity_id."""
        if not entities:
            return

        async with self._driver.session(database=self.database) as session:
            for entity in entities:
                entity_id = _compute_entity_id(entity.name, entity.entity_type, knowledge_id)
                await session.run(
                    _ENTITY_MERGE_QUERY,
                    entity_id=entity_id,
                    name=entity.name,
                    entity_type=entity.entity_type,
                    category=entity.category,
                    value=entity.value,
                    knowledge_id=knowledge_id,
                    source_id=source_id,
                    properties=entity.properties or {},
                )

        logger.info("upserted %d entities for source %s", len(entities), source_id)

    async def add_relations(
        self,
        source_id: str,
        relations: list[GraphRelation],
    ) -> None:
        """Create relationships between existing entity nodes."""
        if not relations:
            return

        async with self._driver.session(database=self.database) as session:
            for rel in relations:
                from_id = _compute_entity_id(rel.from_entity, rel.from_type, rel.knowledge_id)
                to_id = _compute_entity_id(rel.to_entity, rel.to_type, rel.knowledge_id)
                rel_type = _validate_relation_type(rel.relation_type)

                query = (
                    f"MATCH (a:Entity {{entity_id: $from_id}})\n"
                    f"MATCH (b:Entity {{entity_id: $to_id}})\n"
                    f"MERGE (a)-[r:{rel_type} {{source_id: $source_id}}]->(b)\n"
                    f"SET r.context = $context, r.confidence = $confidence"
                )
                await session.run(
                    query,
                    from_id=from_id,
                    to_id=to_id,
                    source_id=source_id,
                    context=rel.context or "",
                    confidence=rel.confidence,
                )

        logger.info("created %d relations for source %s", len(relations), source_id)

    async def query_graph(
        self,
        query: str,
        knowledge_id: str | None = None,
        entity_types: list[str] | None = None,
        max_hops: int = 2,
        top_k: int = 10,
    ) -> list[GraphResult]:
        """Find entities matching the query via full-text index, then traverse N hops."""
        if not query or not query.strip():
            return []

        timeout = self.query_timeout
        async with self._driver.session(database=self.database) as session:
            seeds = await self._find_seed_entities(
                session, query, knowledge_id, entity_types, limit=top_k, timeout=timeout
            )

            if not seeds:
                return []

            results: list[GraphResult] = []
            for seed_node, seed_score in seeds:
                connected_entities, paths = await self._traverse(
                    session, seed_node["entity_id"], max_hops, knowledge_id, timeout=timeout
                )
                results.append(
                    GraphResult(
                        entity=_node_to_entity(seed_node),
                        connected_entities=connected_entities,
                        paths=paths,
                        relevance_score=seed_score,
                    )
                )

        return results[:top_k]

    async def delete_by_source(self, source_id: str) -> None:
        """Remove all graph data introduced by a source, handling dedup correctly.

        Runs as a single transaction so partial failures don't leave inconsistent state.
        """
        async with self._driver.session(database=self.database) as session:
            tx = await session.begin_transaction()
            try:
                await tx.run(_DELETE_RELATIONS_QUERY, source_id=source_id)
                await tx.run(_DELETE_SOURCE_FROM_ENTITIES_QUERY, source_id=source_id)
                await tx.run(_DELETE_ORPHANED_ENTITIES_QUERY)
                await tx.run(_DELETE_PAGES_QUERY, source_id=source_id)
                await tx.run(_DELETE_DOCUMENT_QUERY, source_id=source_id)
                await tx.commit()
            except Exception:
                await tx.rollback()
                raise

        logger.info("deleted graph data for source %s", source_id)

    async def shutdown(self) -> None:
        """Close the Neo4j driver."""
        if self._driver:
            await self._driver.close()
            self._driver = None
            logger.info("neo4j driver closed")

    async def _find_seed_entities(
        self,
        session: Any,
        query: str,
        knowledge_id: str | None,
        entity_types: list[str] | None,
        limit: int,
        timeout: float | None = None,
    ) -> list[tuple[dict[str, Any], float]]:
        """Full-text search for seed entities."""
        if entity_types:
            result = await session.run(
                _SEED_QUERY_WITH_TYPES,
                query_text=query,
                knowledge_id=knowledge_id,
                entity_types=entity_types,
                seed_limit=limit,
                timeout=timeout,
            )
        else:
            result = await session.run(
                _SEED_QUERY,
                query_text=query,
                knowledge_id=knowledge_id,
                seed_limit=limit,
                timeout=timeout,
            )

        seeds: list[tuple[dict[str, Any], float]] = []
        async for record in result:
            node = dict(record["node"])
            score = float(record["score"])
            seeds.append((node, score))

        return seeds

    async def _traverse(
        self,
        session: Any,
        seed_id: str,
        max_hops: int,
        knowledge_id: str | None,
        timeout: float | None = None,
    ) -> tuple[list[GraphEntity], list[GraphPath]]:
        """Walk N hops from a seed entity, return connected entities and paths."""
        hops = min(max(1, max_hops), 5)
        query = _TRAVERSE_QUERY.replace("{max_hops}", str(hops))

        result = await session.run(query, seed_id=seed_id, timeout=timeout)

        seen_entities: dict[str, GraphEntity] = {}
        paths: list[GraphPath] = []

        async for record in result:
            connected = record.get("connected")
            if connected is None:
                continue

            connected_dict = dict(connected)
            entity = _node_to_entity(connected_dict)
            entity_key = f"{entity.name}|{entity.entity_type}"
            if entity_key not in seen_entities:
                seen_entities[entity_key] = entity

            node_names = record.get("node_names", [])
            rel_types = record.get("rel_types", [])
            if node_names and rel_types:
                paths.append(
                    GraphPath(
                        entities=list(node_names),
                        relationships=list(rel_types),
                    )
                )

        return list(seen_entities.values()), paths
