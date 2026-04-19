from types import SimpleNamespace
from unittest.mock import AsyncMock

from rfnry_rag.retrieval.common.models import RetrievedChunk
from rfnry_rag.retrieval.modules.retrieval.search.service import RetrievalService


def _chunk(chunk_id: str, score: float = 0.9) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=chunk_id, source_id="s1", content="text", score=score)


def _make_service() -> RetrievalService:
    mock_vector = SimpleNamespace(
        name="vector",
        weight=1.0,
        top_k=None,
        search=AsyncMock(return_value=[_chunk("v1", 0.9), _chunk("v2", 0.8)]),
    )
    return RetrievalService(retrieval_methods=[mock_vector], top_k=5)


async def test_retrieve_includes_tree_chunks():
    """Pre-computed tree chunks are merged into results via reciprocal rank fusion."""
    service = _make_service()
    tree_chunks = [
        _chunk("tree-root-0", 1.0),
        _chunk("tree-root-1", 1.0),
    ]

    results = await service.retrieve("test query", tree_chunks=tree_chunks)

    result_ids = {r.chunk_id for r in results}
    # Tree chunks should appear alongside vector results
    assert "tree-root-0" in result_ids
    assert "tree-root-1" in result_ids
    # Vector results should also be present
    assert "v1" in result_ids
    assert "v2" in result_ids


async def test_retrieve_without_tree_chunks():
    """When tree_chunks is None (default), retrieval works as before."""
    service = _make_service()

    results = await service.retrieve("test query")

    result_ids = {r.chunk_id for r in results}
    assert "v1" in result_ids
    assert "v2" in result_ids
    assert len(results) == 2
