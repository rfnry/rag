from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rfnry_knowledge.stores.graph.models import GraphEntity, GraphRelation


@pytest.fixture
def mock_driver():
    """Create a mock neo4j AsyncDriver.

    Neo4j's real driver.session() is a sync method that returns an async context manager,
    so we use MagicMock for the session call and set up __aenter__/__aexit__ on the result.
    """
    driver = AsyncMock()
    session = AsyncMock()

    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=False)
    driver.session = MagicMock(return_value=ctx)

    driver.verify_connectivity = AsyncMock()
    driver.close = AsyncMock()
    return driver, session


async def test_compute_entity_id_deterministic():
    from rfnry_knowledge.stores.graph.neo4j import _compute_entity_id

    id1 = _compute_entity_id("Motor M1", "motor", "kb-1")
    id2 = _compute_entity_id("Motor M1", "motor", "kb-1")
    assert id1 == id2
    assert isinstance(id1, str)
    assert len(id1) == 64


async def test_compute_entity_id_case_insensitive():
    from rfnry_knowledge.stores.graph.neo4j import _compute_entity_id

    id1 = _compute_entity_id("Motor M1", "motor", "kb-1")
    id2 = _compute_entity_id("motor m1", "Motor", "kb-1")
    assert id1 == id2


async def test_compute_entity_id_whitespace_collapsed():
    from rfnry_knowledge.stores.graph.neo4j import _compute_entity_id

    id1 = _compute_entity_id("Motor M1", "motor", "kb-1")
    id2 = _compute_entity_id("Motor  M1", "motor", "kb-1")
    assert id1 == id2


async def test_compute_entity_id_different_knowledge_id():
    from rfnry_knowledge.stores.graph.neo4j import _compute_entity_id

    id1 = _compute_entity_id("Motor M1", "motor", "kb-1")
    id2 = _compute_entity_id("Motor M1", "motor", "kb-2")
    assert id1 != id2


async def test_compute_entity_id_none_knowledge_id():
    from rfnry_knowledge.stores.graph.neo4j import _compute_entity_id

    id1 = _compute_entity_id("Motor M1", "motor", None)
    id2 = _compute_entity_id("Motor M1", "motor", None)
    assert id1 == id2


async def test_validate_relation_type_known():
    from rfnry_knowledge.stores.graph.neo4j import _validate_relation_type

    assert _validate_relation_type("CONNECTS_TO") == "CONNECTS_TO"
    assert _validate_relation_type("POWERED_BY") == "POWERED_BY"
    assert _validate_relation_type("CONTROLLED_BY") == "CONTROLLED_BY"
    assert _validate_relation_type("FEEDS") == "FEEDS"
    assert _validate_relation_type("FLOWS_TO") == "FLOWS_TO"
    assert _validate_relation_type("MENTIONS") == "MENTIONS"
    assert _validate_relation_type("REFERENCES") == "REFERENCES"


async def test_validate_relation_type_normalizes():
    from rfnry_knowledge.stores.graph.neo4j import _validate_relation_type

    assert _validate_relation_type("connects to") == "CONNECTS_TO"
    assert _validate_relation_type("powered by") == "POWERED_BY"


async def test_validate_relation_type_unknown_raises():
    from rfnry_knowledge.stores.graph.neo4j import _validate_relation_type

    with pytest.raises(ValueError):
        _validate_relation_type("UNKNOWN_REL")
    with pytest.raises(ValueError):
        _validate_relation_type("")


@patch("rfnry_knowledge.stores.graph.neo4j.AsyncGraphDatabase")
async def test_initialize_creates_driver_and_indexes(mock_neo4j_module, mock_driver):
    from rfnry_knowledge.stores.graph.neo4j import Neo4jGraphStore

    driver, session = mock_driver
    mock_neo4j_module.driver.return_value = driver

    store = Neo4jGraphStore(uri="bolt://localhost:7687", username="neo4j", password="pass")
    await store.initialize()

    mock_neo4j_module.driver.assert_called_once_with(
        "bolt://localhost:7687",
        auth=("neo4j", "pass"),
        max_connection_pool_size=10,
        connection_acquisition_timeout=5.0,
        connection_timeout=5.0,
    )
    driver.verify_connectivity.assert_called_once()
    assert session.run.call_count > 0


@patch("rfnry_knowledge.stores.graph.neo4j.AsyncGraphDatabase")
async def test_add_entities_calls_merge(mock_neo4j_module, mock_driver):
    from rfnry_knowledge.stores.graph.neo4j import Neo4jGraphStore

    driver, session = mock_driver
    mock_neo4j_module.driver.return_value = driver

    store = Neo4jGraphStore(uri="bolt://localhost:7687", password="test-pw")
    await store.initialize()

    entities = [
        GraphEntity(name="Motor M1", entity_type="motor", category="electrical", value="480V"),
        GraphEntity(name="Breaker CB-3", entity_type="breaker", category="electrical"),
    ]
    await store.add_entities(source_id="src-1", knowledge_id="kb-1", entities=entities)

    merge_calls = [c for c in session.run.call_args_list if "MERGE" in str(c)]
    assert len(merge_calls) >= 2


@patch("rfnry_knowledge.stores.graph.neo4j.AsyncGraphDatabase")
async def test_add_relations_calls_match_and_merge(mock_neo4j_module, mock_driver):
    from rfnry_knowledge.stores.graph.neo4j import Neo4jGraphStore

    driver, session = mock_driver
    mock_neo4j_module.driver.return_value = driver

    store = Neo4jGraphStore(uri="bolt://localhost:7687", password="test-pw")
    await store.initialize()

    relations = [
        GraphRelation(
            from_entity="Motor M1",
            from_type="motor",
            to_entity="Breaker CB-3",
            to_type="breaker",
            relation_type="POWERED_BY",
            knowledge_id="kb-1",
            context="power feed",
        ),
    ]
    await store.add_relations(source_id="src-1", relations=relations)

    rel_calls = [c for c in session.run.call_args_list if "MATCH" in str(c) and "POWERED_BY" in str(c)]
    assert len(rel_calls) >= 1


@patch("rfnry_knowledge.stores.graph.neo4j.AsyncGraphDatabase")
async def test_delete_by_source_runs_cleanup_in_transaction(mock_neo4j_module, mock_driver):
    from rfnry_knowledge.stores.graph.neo4j import Neo4jGraphStore

    driver, session = mock_driver
    mock_neo4j_module.driver.return_value = driver

    # The store now uses `async with await session.begin_transaction() as tx:`
    # for automatic commit/rollback, so the test exercises the tx via its
    # context-manager protocol rather than explicit commit/rollback calls.
    tx = AsyncMock()
    tx.run = AsyncMock()
    # AsyncMock's default __aenter__ returns a fresh mock; bind it to tx so the
    # `async with ... as tx:` body uses the same object we track.
    tx.__aenter__.return_value = tx
    session.begin_transaction = AsyncMock(return_value=tx)

    store = Neo4jGraphStore(uri="bolt://localhost:7687", password="test-pw")
    await store.initialize()

    await store.delete_by_source(source_id="src-1")

    assert tx.run.call_count == 5
    tx.__aenter__.assert_awaited_once()
    tx.__aexit__.assert_awaited_once()


@patch("rfnry_knowledge.stores.graph.neo4j.AsyncGraphDatabase")
async def test_query_graph_returns_empty_on_no_seeds(mock_neo4j_module, mock_driver):
    from rfnry_knowledge.stores.graph.neo4j import Neo4jGraphStore

    driver, session = mock_driver
    mock_neo4j_module.driver.return_value = driver

    class EmptyAsyncIter:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    session.run.return_value = EmptyAsyncIter()

    store = Neo4jGraphStore(uri="bolt://localhost:7687", password="test-pw")
    await store.initialize()

    results = await store.query_graph(query="nonexistent entity", knowledge_id="kb-1")
    assert results == []


@patch("rfnry_knowledge.stores.graph.neo4j.AsyncGraphDatabase")
async def test_shutdown_closes_driver(mock_neo4j_module, mock_driver):
    from rfnry_knowledge.stores.graph.neo4j import Neo4jGraphStore

    driver, session = mock_driver
    mock_neo4j_module.driver.return_value = driver

    store = Neo4jGraphStore(uri="bolt://localhost:7687", password="test-pw")
    await store.initialize()
    await store.shutdown()

    driver.close.assert_called_once()


@patch("rfnry_knowledge.stores.graph.neo4j.AsyncGraphDatabase")
async def test_shutdown_noop_when_not_initialized(mock_neo4j_module):
    from rfnry_knowledge.stores.graph.neo4j import Neo4jGraphStore

    store = Neo4jGraphStore(uri="bolt://localhost:7687", password="test-pw")
    await store.shutdown()


async def test_add_entities_empty_list_is_noop():
    from unittest.mock import patch as _patch

    from rfnry_knowledge.stores.graph.neo4j import Neo4jGraphStore

    with _patch("rfnry_knowledge.stores.graph.neo4j.AsyncGraphDatabase") as mock_neo4j_module:
        driver = AsyncMock()
        session = AsyncMock()
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=session)
        ctx.__aexit__ = AsyncMock(return_value=False)
        driver.session = MagicMock(return_value=ctx)
        driver.verify_connectivity = AsyncMock()
        mock_neo4j_module.driver.return_value = driver

        store = Neo4jGraphStore(uri="bolt://localhost:7687", password="test-pw")
        await store.initialize()

        init_calls = session.run.call_count
        await store.add_entities(source_id="src-1", knowledge_id="kb-1", entities=[])
        assert session.run.call_count == init_calls
