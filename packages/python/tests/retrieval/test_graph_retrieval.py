# src/rfnry-rag/retrieval/tests/test_graph_retrieval_method.py
from types import SimpleNamespace
from unittest.mock import AsyncMock

from rfnry_rag.retrieval.common.models import RetrievedChunk
from rfnry_rag.retrieval.methods.graph import GraphRetrieval
from rfnry_rag.retrieval.search.service import RetrievalService
from rfnry_rag.stores.graph.models import GraphEntity, GraphPath, GraphResult


def _make_graph_results():
    return [
        GraphResult(
            entity=GraphEntity(
                name="Motor M1",
                entity_type="motor",
                category="electrical",
                value="480V 3-phase",
                properties={"source_id": "src-2"},
            ),
            connected_entities=[
                GraphEntity(name="Breaker CB-3", entity_type="breaker", category="electrical"),
                GraphEntity(name="VFD-3", entity_type="vfd", category="electrical"),
            ],
            paths=[
                GraphPath(
                    entities=["Motor M1", "Breaker CB-3", "Panel MCC-1"],
                    relationships=["POWERED_BY", "FEEDS"],
                ),
            ],
            relevance_score=0.95,
        ),
    ]


def _make_graph_method(graph_results):
    """Build a mock graph retrieval method that returns pre-converted chunks."""
    chunks = GraphRetrieval._convert(graph_results)
    return SimpleNamespace(
        name="graph",
        weight=1.0,
        top_k=None,
        search=AsyncMock(return_value=chunks),
    )


def _make_service(graph_method=None, document_method=None):
    mock_vector = SimpleNamespace(
        name="vector",
        weight=1.0,
        top_k=None,
        search=AsyncMock(
            return_value=[
                RetrievedChunk(chunk_id="chunk-1", source_id="src-1", content="Vector result", score=0.8),
            ]
        ),
    )
    methods = [mock_vector]
    if document_method is not None:
        methods.append(document_method)
    if graph_method is not None:
        methods.append(graph_method)
    return RetrievalService(
        retrieval_methods=methods,
        reranking=None,
        top_k=5,
    )


async def test_search_converts_graph_results():
    store = AsyncMock()
    store.query_graph = AsyncMock(return_value=_make_graph_results())

    method = GraphRetrieval(graph_store=store, weight=0.7)
    assert method.name == "graph"
    assert method.weight == 0.7

    results = await method.search(query="Motor M1", top_k=5, knowledge_id="kb-1")
    assert len(results) == 1
    assert results[0].chunk_id == "graph:Motor M1:motor"
    assert results[0].score == 0.95
    assert "480V 3-phase" in results[0].content
    assert "POWERED_BY" in results[0].content
    assert results[0].source_metadata["retrieval_type"] == "graph"
    store.query_graph.assert_called_once_with(query="Motor M1", knowledge_id="kb-1", max_hops=2, top_k=5)


async def test_search_empty():
    store = AsyncMock()
    store.query_graph = AsyncMock(return_value=[])

    method = GraphRetrieval(graph_store=store)
    results = await method.search(query="nothing", top_k=5)
    assert results == []


async def test_error_returns_empty():
    store = AsyncMock()
    store.query_graph = AsyncMock(side_effect=RuntimeError("neo4j down"))

    method = GraphRetrieval(graph_store=store)
    results = await method.search(query="test", top_k=5)
    assert results == []


async def test_graph_results_to_chunks_no_value():
    results = [
        GraphResult(
            entity=GraphEntity(name="Pump P-1", entity_type="pump", category="mechanical", properties={}),
            connected_entities=[],
            paths=[],
            relevance_score=0.5,
        ),
    ]
    chunks = GraphRetrieval._convert(results)

    assert len(chunks) == 1
    assert "Specifications:" not in chunks[0].content
    assert chunks[0].chunk_id == "graph:Pump P-1:pump"


# --- Integration tests (RetrievalService with graph method) ---


async def test_retrieve_with_graph_store():
    graph_results = _make_graph_results()
    mock_graph = _make_graph_method(graph_results)

    service = _make_service(graph_method=mock_graph)
    results, _ = await service.retrieve(query="what connects to Motor M1?", knowledge_id="kb-1")

    mock_graph.search.assert_called_once()
    assert len(results) >= 2
    graph_chunks = [r for r in results if r.chunk_id.startswith("graph:")]
    assert len(graph_chunks) == 1


async def test_retrieve_without_graph_store():
    service = _make_service(graph_method=None)
    results, _ = await service.retrieve(query="test query", knowledge_id="kb-1")

    assert len(results) == 1
    assert results[0].chunk_id == "chunk-1"


async def test_graph_store_empty_result_no_fusion():
    mock_graph = SimpleNamespace(
        name="graph",
        weight=1.0,
        top_k=None,
        search=AsyncMock(return_value=[]),
    )

    service = _make_service(graph_method=mock_graph)
    results, _ = await service.retrieve(query="test query", knowledge_id="kb-1")

    assert len(results) == 1
    assert results[0].chunk_id == "chunk-1"


async def test_graph_and_document_store_together():
    graph_results = _make_graph_results()
    mock_graph = _make_graph_method(graph_results)

    mock_document = SimpleNamespace(
        name="document",
        weight=1.0,
        top_k=None,
        search=AsyncMock(
            return_value=[
                RetrievedChunk(
                    chunk_id="fulltext:src-3",
                    source_id="src-3",
                    content="excerpt",
                    score=0.7,
                    source_metadata={"title": "Doc", "match_type": "fulltext"},
                ),
            ]
        ),
    )

    service = _make_service(graph_method=mock_graph, document_method=mock_document)
    results, _ = await service.retrieve(query="Motor M1", knowledge_id="kb-1")

    assert len(results) == 3
