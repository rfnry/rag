"""GraphRetrieval.trace: N-hop traversal returning GraphPath objects."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from rfnry_rag.retrieval.methods.graph import GraphRetrieval
from rfnry_rag.stores.graph.models import GraphEntity, GraphPath, GraphResult


def _result(seed_name: str, paths: list[GraphPath]) -> GraphResult:
    return GraphResult(
        entity=GraphEntity(name=seed_name, entity_type="component"),
        connected_entities=[],
        paths=paths,
        relevance_score=1.0,
    )


@pytest.mark.asyncio
async def test_trace_returns_paths_from_seed() -> None:
    p1 = GraphPath(entities=["V-101", "P-201"], relationships=["CONNECTS_TO"], description="1-hop")
    p2 = GraphPath(entities=["V-101", "P-201", "T-301"], relationships=["CONNECTS_TO", "FLOWS_TO"], description="2-hop")
    mock_store = SimpleNamespace(query_graph=AsyncMock(return_value=[_result("V-101", [p1, p2])]))
    gr = GraphRetrieval(store=mock_store)
    paths = await gr.trace(entity_name="V-101", max_hops=3)
    assert len(paths) == 2
    assert paths[0].entities == ["V-101", "P-201"]


@pytest.mark.asyncio
async def test_trace_filters_by_relation_types() -> None:
    p1 = GraphPath(entities=["V-101", "P-201"], relationships=["CONNECTS_TO"])
    p2 = GraphPath(entities=["V-101", "T-301"], relationships=["FLOWS_TO"])
    p3 = GraphPath(entities=["V-101", "P-201", "T-301"], relationships=["CONNECTS_TO", "FLOWS_TO"])
    mock_store = SimpleNamespace(query_graph=AsyncMock(return_value=[_result("V-101", [p1, p2, p3])]))
    gr = GraphRetrieval(store=mock_store)
    paths = await gr.trace(entity_name="V-101", max_hops=3, relation_types=["FLOWS_TO"])
    assert len(paths) == 1
    assert paths[0].relationships == ["FLOWS_TO"]


@pytest.mark.asyncio
async def test_trace_respects_max_hops() -> None:
    p1 = GraphPath(entities=["A", "B"], relationships=["CONNECTS_TO"])
    mock_store = SimpleNamespace(query_graph=AsyncMock(return_value=[_result("A", [p1])]))
    gr = GraphRetrieval(store=mock_store)
    paths = await gr.trace(entity_name="A", max_hops=1)
    assert mock_store.query_graph.await_args.kwargs["max_hops"] == 1
    for p in paths:
        assert len(p.relationships) <= 1


@pytest.mark.asyncio
async def test_trace_empty_when_no_seed_matches() -> None:
    mock_store = SimpleNamespace(query_graph=AsyncMock(return_value=[]))
    gr = GraphRetrieval(store=mock_store)
    paths = await gr.trace(entity_name="DOES_NOT_EXIST", max_hops=2)
    assert paths == []


@pytest.mark.asyncio
async def test_trace_propagates_knowledge_id_filter() -> None:
    mock_store = SimpleNamespace(query_graph=AsyncMock(return_value=[]))
    gr = GraphRetrieval(store=mock_store)
    await gr.trace(entity_name="X", max_hops=2, knowledge_id="k1")
    kwargs = mock_store.query_graph.await_args.kwargs
    assert kwargs.get("knowledge_id") == "k1"


@pytest.mark.asyncio
async def test_trace_store_error_returns_empty_list() -> None:
    """Same defensive error behavior as search()."""
    mock_store = SimpleNamespace(query_graph=AsyncMock(side_effect=RuntimeError("boom")))
    gr = GraphRetrieval(store=mock_store)
    paths = await gr.trace(entity_name="X", max_hops=2)
    assert paths == []


@pytest.mark.asyncio
async def test_trace_aggregates_paths_across_multiple_seeds() -> None:
    """If the seed lookup returns multiple results, all paths are aggregated."""
    p1 = GraphPath(entities=["V-101", "P-201"], relationships=["CONNECTS_TO"])
    p2 = GraphPath(entities=["V-102", "T-301"], relationships=["FLOWS_TO"])
    mock_store = SimpleNamespace(query_graph=AsyncMock(return_value=[_result("V-101", [p1]), _result("V-102", [p2])]))
    gr = GraphRetrieval(store=mock_store)
    paths = await gr.trace(entity_name="V", max_hops=2)
    assert len(paths) == 2
