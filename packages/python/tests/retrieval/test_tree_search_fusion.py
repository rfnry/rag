from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from rfnry_rag.retrieval.common.models import RetrievedChunk


def _chunk(cid: str, score: float = 1.0) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=cid, source_id="s1", content=cid, score=score)


def _engine(make_engine: Any, *, metadata_store: Any, tree_search_service: Any) -> Any:
    """Build a barebones tree-search engine via the shared ``make_engine`` factory.

    Tree-search tests need only ``persistence.metadata_store`` and the
    ``_tree_search_service`` attribute — pass an explicit ``SimpleNamespace``
    config to bypass the factory's MagicMock(spec=RagServerConfig) shape.
    """
    return make_engine(
        config=SimpleNamespace(persistence=SimpleNamespace(metadata_store=metadata_store)),
        tree_search_service=tree_search_service,
    )


@pytest.mark.asyncio
async def test_tree_chunks_threaded_into_single_unstructured_call(
    make_engine: Any,
) -> None:
    """When tree search returns chunks, they must be fused via RetrievalService
    (threaded through the `tree_chunks` kwarg), not by re-invoking retrieve()
    and discarding the first result set."""
    engine = _engine(make_engine, metadata_store=object(), tree_search_service=object())

    unstructured = SimpleNamespace(retrieve=AsyncMock(return_value=([_chunk("u1"), _chunk("u2"), _chunk("t1")], None)))
    engine._get_retrieval = cast(Any, lambda _c: (unstructured, None))  # type: ignore[method-assign]
    engine._build_retrieval_query = cast(Any, lambda text, history: text)  # type: ignore[method-assign]
    engine._run_tree_search = cast(Any, AsyncMock(return_value=[_chunk("t1")]))  # type: ignore[method-assign]

    chunks, _ = await engine._retrieve_chunks(
        text="q", history=None, knowledge_id=None, min_score=None, collection="default"
    )

    assert unstructured.retrieve.await_count == 1
    call = unstructured.retrieve.await_args
    assert call.kwargs.get("tree_chunks") == [_chunk("t1")]
    assert len(chunks) == 3


@pytest.mark.asyncio
async def test_no_tree_chunks_passes_no_tree_kwarg(make_engine: Any) -> None:
    """When tree search returns nothing, the tree_chunks kwarg is omitted."""
    engine = _engine(make_engine, metadata_store=object(), tree_search_service=object())

    unstructured = SimpleNamespace(retrieve=AsyncMock(return_value=([_chunk("u1")], None)))
    engine._get_retrieval = cast(Any, lambda _c: (unstructured, None))  # type: ignore[method-assign]
    engine._build_retrieval_query = cast(Any, lambda text, history: text)  # type: ignore[method-assign]
    engine._run_tree_search = cast(Any, AsyncMock(return_value=[]))  # type: ignore[method-assign]

    await engine._retrieve_chunks(text="q", history=None, knowledge_id=None, min_score=None, collection="default")

    assert unstructured.retrieve.await_count == 1
    call = unstructured.retrieve.await_args
    assert "tree_chunks" not in call.kwargs


@pytest.mark.asyncio
async def test_tree_search_without_service_skips_tree_path(make_engine: Any) -> None:
    engine = _engine(make_engine, metadata_store=None, tree_search_service=None)

    unstructured = SimpleNamespace(retrieve=AsyncMock(return_value=([_chunk("u1")], None)))
    engine._get_retrieval = cast(Any, lambda _c: (unstructured, None))  # type: ignore[method-assign]
    engine._build_retrieval_query = cast(Any, lambda text, history: text)  # type: ignore[method-assign]

    chunks, _ = await engine._retrieve_chunks(
        text="q", history=None, knowledge_id=None, min_score=None, collection="default"
    )

    assert unstructured.retrieve.await_count == 1
    assert len(chunks) == 1
