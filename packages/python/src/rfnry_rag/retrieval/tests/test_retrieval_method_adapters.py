"""Adapter tests — TreeRetrieval and StructuredRetrieval wrap internal services
so they conform to the BaseRetrievalMethod protocol and can be composed into
a RetrievalService."""

import json
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest

from rfnry_rag.retrieval.common.models import RetrievedChunk, Source, TreeIndex
from rfnry_rag.retrieval.modules.retrieval.methods.enrich import StructuredRetrieval
from rfnry_rag.retrieval.modules.retrieval.methods.tree import TreeRetrieval


def _chunk(cid: str) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=cid, source_id="s", content=cid, score=1.0)


def _assert_conforms_to_protocol(adapter: object, expected_name: str) -> None:
    """Structural-conformance helper — BaseRetrievalMethod is not runtime_checkable."""
    assert hasattr(adapter, "name")
    assert hasattr(adapter, "weight")
    assert hasattr(adapter, "top_k")
    assert callable(getattr(adapter, "search", None))
    assert adapter.name == expected_name  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_structured_retrieval_adapter_conforms_to_protocol() -> None:
    service = SimpleNamespace(retrieve=AsyncMock(return_value=[_chunk("c1")]))
    adapter = StructuredRetrieval(service=cast(object, service), weight=0.7, top_k=3)  # type: ignore[arg-type]

    _assert_conforms_to_protocol(adapter, "enrich")
    assert adapter.weight == 0.7
    assert adapter.top_k == 3

    result = await adapter.search("query", top_k=5, knowledge_id="k1")
    assert result == [_chunk("c1")]

    service.retrieve.assert_awaited_once_with(query="query", knowledge_id="k1", top_k=3)


@pytest.mark.asyncio
async def test_structured_retrieval_uses_passed_top_k_when_none_configured() -> None:
    service = SimpleNamespace(retrieve=AsyncMock(return_value=[]))
    adapter = StructuredRetrieval(service=cast(object, service))  # type: ignore[arg-type]
    await adapter.search("q", top_k=10)
    service.retrieve.assert_awaited_once_with(query="q", knowledge_id=None, top_k=10)


@pytest.mark.asyncio
async def test_structured_retrieval_isolates_errors() -> None:
    service = SimpleNamespace(retrieve=AsyncMock(side_effect=RuntimeError("boom")))
    adapter = StructuredRetrieval(service=cast(object, service))  # type: ignore[arg-type]
    result = await adapter.search("q", top_k=5)
    assert result == []


def _source(sid: str) -> Source:
    return Source(source_id=sid, knowledge_id="k", source_type="manual", status="completed")


@pytest.mark.asyncio
async def test_tree_retrieval_adapter_conforms_to_protocol() -> None:
    tree_index = TreeIndex(
        source_id="s1",
        doc_name="doc",
        doc_description=None,
        structure=[],
        page_count=0,
        created_at=datetime.now(UTC),
        pages=[],
    )
    # Empty pages means the loop skips; no _service.search call. Test conformance only.
    metadata_store = SimpleNamespace(
        list_sources=AsyncMock(return_value=[_source("s1")]),
        get_tree_index=AsyncMock(return_value=json.dumps(tree_index.to_dict())),
    )
    service = SimpleNamespace(search=AsyncMock(return_value=[]), to_retrieved_chunks=lambda r, t: [])

    adapter = TreeRetrieval(
        service=cast(object, service),  # type: ignore[arg-type]
        metadata_store=cast(object, metadata_store),  # type: ignore[arg-type]
        weight=0.9,
    )
    _assert_conforms_to_protocol(adapter, "tree")
    assert adapter.weight == 0.9
    assert adapter.top_k is None

    result = await adapter.search("q", top_k=5, knowledge_id="k")
    assert result == []
    metadata_store.list_sources.assert_awaited_once_with(knowledge_id="k")
    # tree_index.pages is empty → service.search not invoked
    service.search.assert_not_called()


@pytest.mark.asyncio
async def test_tree_retrieval_isolates_errors() -> None:
    metadata_store = SimpleNamespace(list_sources=AsyncMock(side_effect=RuntimeError("boom")))
    service = SimpleNamespace()
    adapter = TreeRetrieval(
        service=cast(object, service),  # type: ignore[arg-type]
        metadata_store=cast(object, metadata_store),  # type: ignore[arg-type]
    )
    result = await adapter.search("q", top_k=5)
    assert result == []
