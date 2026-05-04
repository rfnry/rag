from rfnry_knowledge.stores.graph.neo4j import Neo4jGraphStore


def test_default_label_prefix_is_empty() -> None:
    store = Neo4jGraphStore(uri="bolt://localhost:7687", password="x")
    assert store.entity_label == "Entity"
    assert store.fulltext_index_name == "entity_search"


def test_memory_label_prefix_applied() -> None:
    store = Neo4jGraphStore(
        uri="bolt://localhost:7687", password="x", node_label_prefix="Memory",
    )
    assert store.entity_label == "MemoryEntity"
    assert store.fulltext_index_name == "memory_entity_search"


def test_prefixed_constraint_names_avoid_collision() -> None:
    a = Neo4jGraphStore(uri="bolt://localhost:7687", password="x")
    b = Neo4jGraphStore(
        uri="bolt://localhost:7687", password="x", node_label_prefix="Memory",
    )
    # Constraint queries differ — at minimum the entity_id constraint name does.
    a_index = a._index_queries[0]
    b_index = b._index_queries[0]
    assert a_index != b_index
    assert "MemoryEntity" in b_index
    assert "Entity" in a_index and "MemoryEntity" not in a_index
