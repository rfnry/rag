"""Adapter tests — StructuredRetrieval wraps an internal service so it conforms
to the BaseRetrievalMethod protocol and can be composed into a RetrievalService."""

from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest

from rfnry_rag.retrieval.common.models import RetrievedChunk
from rfnry_rag.retrieval.methods.enrich import StructuredRetrieval


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
