from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from rfnry_rag.retrieval.common.models import RetrievedChunk
from rfnry_rag.retrieval.server import RagEngine


def _chunk(cid: str, score: float = 1.0) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=cid, source_id="s1", content=cid, score=score)


def _make_engine(*, metadata_store: Any, tree_search_service: Any) -> RagEngine:
    engine = RagEngine.__new__(RagEngine)
    engine._config = cast(Any, SimpleNamespace(persistence=SimpleNamespace(metadata_store=metadata_store)))
    engine._tree_search_service = tree_search_service
    return engine


@pytest.mark.asyncio
async def test_tree_chunks_threaded_into_single_unstructured_call() -> None:
    """When tree search returns chunks, they must be fused via RetrievalService
    (threaded through the `tree_chunks` kwarg), not by re-invoking retrieve()
    and discarding the first result set."""
    engine = _make_engine(metadata_store=object(), tree_search_service=object())

    unstructured = SimpleNamespace(
        retrieve=AsyncMock(return_value=[_chunk("u1"), _chunk("u2"), _chunk("t1")])
    )
    engine._get_retrieval = cast(Any, lambda _c: (unstructured, None))  # type: ignore[method-assign]
    engine._build_retrieval_query = cast(Any, lambda text, history: text)  # type: ignore[method-assign]
    engine._run_tree_search = cast(Any, AsyncMock(return_value=[_chunk("t1")]))  # type: ignore[method-assign]

    chunks = await engine._retrieve_chunks(
        text="q", history=None, knowledge_id=None, min_score=None, collection="default"
    )

    assert unstructured.retrieve.await_count == 1
    call = unstructured.retrieve.await_args
    assert call.kwargs.get("tree_chunks") == [_chunk("t1")]
    assert len(chunks) == 3


@pytest.mark.asyncio
async def test_no_tree_chunks_passes_no_tree_kwarg() -> None:
    """When tree search returns nothing, the tree_chunks kwarg is omitted."""
    engine = _make_engine(metadata_store=object(), tree_search_service=object())

    unstructured = SimpleNamespace(retrieve=AsyncMock(return_value=[_chunk("u1")]))
    engine._get_retrieval = cast(Any, lambda _c: (unstructured, None))  # type: ignore[method-assign]
    engine._build_retrieval_query = cast(Any, lambda text, history: text)  # type: ignore[method-assign]
    engine._run_tree_search = cast(Any, AsyncMock(return_value=[]))  # type: ignore[method-assign]

    await engine._retrieve_chunks(
        text="q", history=None, knowledge_id=None, min_score=None, collection="default"
    )

    assert unstructured.retrieve.await_count == 1
    call = unstructured.retrieve.await_args
    assert "tree_chunks" not in call.kwargs


@pytest.mark.asyncio
async def test_tree_search_without_service_skips_tree_path() -> None:
    engine = _make_engine(metadata_store=None, tree_search_service=None)

    unstructured = SimpleNamespace(retrieve=AsyncMock(return_value=[_chunk("u1")]))
    engine._get_retrieval = cast(Any, lambda _c: (unstructured, None))  # type: ignore[method-assign]
    engine._build_retrieval_query = cast(Any, lambda text, history: text)  # type: ignore[method-assign]

    chunks = await engine._retrieve_chunks(
        text="q", history=None, knowledge_id=None, min_score=None, collection="default"
    )

    assert unstructured.retrieve.await_count == 1
    assert len(chunks) == 1
