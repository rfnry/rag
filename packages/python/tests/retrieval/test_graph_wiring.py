from unittest.mock import AsyncMock

from rfnry_rag.retrieval.server import PersistenceConfig


def _mock_vector_store():
    m = AsyncMock()
    m.initialize = AsyncMock()
    m.shutdown = AsyncMock()
    return m


def _mock_graph_store():
    m = AsyncMock()
    m.initialize = AsyncMock()
    m.shutdown = AsyncMock()
    m.add_entities = AsyncMock()
    m.add_relations = AsyncMock()
    m.query_graph = AsyncMock(return_value=[])
    m.delete_by_source = AsyncMock()
    return m


def test_persistence_config_accepts_graph_store():
    vector_store = _mock_vector_store()
    graph_store = _mock_graph_store()
    config = PersistenceConfig(vector_store=vector_store, graph_store=graph_store)
    assert config.graph_store is graph_store


def test_persistence_config_graph_store_defaults_none():
    vector_store = _mock_vector_store()
    config = PersistenceConfig(vector_store=vector_store)
    assert config.graph_store is None


async def test_knowledge_manager_remove_calls_graph_store():
    from rfnry_rag.retrieval.common.models import Source
    from rfnry_rag.retrieval.modules.knowledge.manager import KnowledgeManager

    vector_store = _mock_vector_store()
    vector_store.delete = AsyncMock(return_value=5)

    metadata_store = AsyncMock()
    metadata_store.get_source = AsyncMock(return_value=Source(source_id="src-1", knowledge_id="kb-1"))
    metadata_store.delete_source = AsyncMock()

    graph_store = _mock_graph_store()

    manager = KnowledgeManager(
        vector_store=vector_store,
        metadata_store=metadata_store,
        graph_store=graph_store,
    )
    deleted = await manager.remove("src-1")

    assert deleted == 5
    graph_store.delete_by_source.assert_called_once_with("src-1")


async def test_knowledge_manager_remove_without_graph_store():
    from rfnry_rag.retrieval.common.models import Source
    from rfnry_rag.retrieval.modules.knowledge.manager import KnowledgeManager

    vector_store = _mock_vector_store()
    vector_store.delete = AsyncMock(return_value=3)

    metadata_store = AsyncMock()
    metadata_store.get_source = AsyncMock(return_value=Source(source_id="src-1", knowledge_id="kb-1"))
    metadata_store.delete_source = AsyncMock()

    manager = KnowledgeManager(
        vector_store=vector_store,
        metadata_store=metadata_store,
    )
    deleted = await manager.remove("src-1")
    assert deleted == 3
