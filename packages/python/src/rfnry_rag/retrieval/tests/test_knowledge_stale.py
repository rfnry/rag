"""KnowledgeManager stale-source API — surface and purge sources whose stored
embedding model differs from the configured one."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from rfnry_rag.retrieval.common.models import Source
from rfnry_rag.retrieval.modules.knowledge.manager import KnowledgeManager


def _source(sid: str, stale: bool = False) -> Source:
    return Source(source_id=sid, knowledge_id="k", status="completed", stale=stale)


@pytest.mark.asyncio
async def test_list_stale_returns_only_stale_sources() -> None:
    metadata_store = SimpleNamespace(
        list_sources=AsyncMock(
            return_value=[_source("a", stale=False), _source("b", stale=True), _source("c", stale=True)]
        )
    )
    km = KnowledgeManager(metadata_store=metadata_store)  # type: ignore[arg-type]
    result = await km.list_stale()
    assert [s.source_id for s in result] == ["b", "c"]


@pytest.mark.asyncio
async def test_list_stale_without_metadata_store_returns_empty() -> None:
    km = KnowledgeManager()
    assert await km.list_stale() == []


@pytest.mark.asyncio
async def test_purge_stale_removes_and_returns_count() -> None:
    stale_sources = [_source("b", stale=True), _source("c", stale=True)]
    metadata_store = SimpleNamespace(
        list_sources=AsyncMock(return_value=[_source("a", stale=False), *stale_sources]),
        get_source=AsyncMock(side_effect=lambda sid: next(s for s in stale_sources if s.source_id == sid)),
        delete_source=AsyncMock(),
    )
    vector_store = SimpleNamespace(delete=AsyncMock(return_value=3))

    km = KnowledgeManager(
        metadata_store=metadata_store,  # type: ignore[arg-type]
        vector_store=vector_store,  # type: ignore[arg-type]
    )
    count = await km.purge_stale()
    assert count == 2
    # delete_source called for each stale source
    assert metadata_store.delete_source.await_count == 2


@pytest.mark.asyncio
async def test_purge_stale_with_no_stale_is_zero() -> None:
    metadata_store = SimpleNamespace(
        list_sources=AsyncMock(return_value=[_source("a", stale=False)]),
        delete_source=AsyncMock(),
    )
    km = KnowledgeManager(metadata_store=metadata_store)  # type: ignore[arg-type]
    assert await km.purge_stale() == 0
    metadata_store.delete_source.assert_not_called()
