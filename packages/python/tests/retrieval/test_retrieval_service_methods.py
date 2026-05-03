from types import SimpleNamespace
from unittest.mock import AsyncMock

from rfnry_knowledge.models import RetrievedChunk
from rfnry_knowledge.retrieval.search.service import RetrievalService


def _mock_method(name: str, results: list[RetrievedChunk]) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        weight=1.0,
        top_k=None,
        search=AsyncMock(return_value=results),
    )


def _mock_method_full(
    name: str, results: list[RetrievedChunk], weight: float = 1.0, top_k: int | None = None
) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        weight=weight,
        top_k=top_k,
        search=AsyncMock(return_value=results),
    )


async def test_dispatch_single_method():
    vector = _mock_method(
        "vector",
        [
            RetrievedChunk(chunk_id="c1", source_id="s1", content="text", score=0.9),
        ],
    )
    service = RetrievalService(retrieval_methods=[vector], top_k=5)
    results, _ = await service.retrieve(query="test", knowledge_id="kb-1")
    assert len(results) == 1
    assert results[0].chunk_id == "c1"
    vector.search.assert_called_once()
    call_kwargs = vector.search.call_args.kwargs
    assert call_kwargs["query"] == "test"
    assert call_kwargs["top_k"] == 20  # fetch_k = 5 * 4
    assert call_kwargs["filters"] == {"knowledge_id": "kb-1"}
    assert call_kwargs["knowledge_id"] == "kb-1"


async def test_dispatch_multiple_methods_fused():
    vector = _mock_method(
        "vector",
        [
            RetrievedChunk(chunk_id="c1", source_id="s1", content="vector", score=0.9),
        ],
    )
    document = _mock_method(
        "document",
        [
            RetrievedChunk(chunk_id="c2", source_id="s2", content="doc", score=0.8),
        ],
    )
    service = RetrievalService(retrieval_methods=[vector, document], top_k=5)
    results, _ = await service.retrieve(query="test")
    assert len(results) == 2
    ids = {r.chunk_id for r in results}
    assert "c1" in ids
    assert "c2" in ids


async def test_empty_method_list():
    service = RetrievalService(retrieval_methods=[], top_k=5)
    results, _ = await service.retrieve(query="test")
    assert results == []


async def test_failed_method_returns_empty_others_succeed():
    vector = _mock_method(
        "vector",
        [
            RetrievedChunk(chunk_id="c1", source_id="s1", content="text", score=0.9),
        ],
    )
    graph = _mock_method("graph", [])
    service = RetrievalService(retrieval_methods=[vector, graph], top_k=5)
    results, _ = await service.retrieve(query="test")
    assert len(results) == 1


async def test_reranker_applied():
    vector = _mock_method(
        "vector",
        [
            RetrievedChunk(chunk_id="c1", source_id="s1", content="text", score=0.5),
            RetrievedChunk(chunk_id="c2", source_id="s2", content="text2", score=0.3),
        ],
    )
    reranker = AsyncMock()
    reranker.rerank = AsyncMock(
        return_value=[
            RetrievedChunk(chunk_id="c2", source_id="s2", content="text2", score=0.95),
        ]
    )
    service = RetrievalService(retrieval_methods=[vector], reranking=reranker, top_k=1)
    results, _ = await service.retrieve(query="test")
    assert len(results) == 1
    assert results[0].chunk_id == "c2"


async def test_method_weight_affects_fusion_scores():
    high_weight = _mock_method_full(
        "vector",
        [
            RetrievedChunk(chunk_id="v1", source_id="s1", content="text", score=0.9),
        ],
        weight=2.0,
    )
    low_weight = _mock_method_full(
        "document",
        [
            RetrievedChunk(chunk_id="d1", source_id="s2", content="doc", score=0.9),
        ],
        weight=0.5,
    )

    service = RetrievalService(retrieval_methods=[high_weight, low_weight], top_k=5)
    results, _ = await service.retrieve(query="test")

    scores = {r.chunk_id: r.score for r in results}
    assert scores["v1"] > scores["d1"]


async def test_method_top_k_override():
    vector = _mock_method_full(
        "vector",
        [
            RetrievedChunk(chunk_id="c1", source_id="s1", content="text", score=0.9),
        ],
        top_k=50,
    )
    document = _mock_method_full(
        "document",
        [
            RetrievedChunk(chunk_id="c2", source_id="s2", content="doc", score=0.8),
        ],
        top_k=None,
    )

    service = RetrievalService(retrieval_methods=[vector, document], top_k=5)
    await service.retrieve(query="test")

    vector_call = vector.search.call_args
    assert vector_call.kwargs["top_k"] == 50

    doc_call = document.search.call_args
    assert doc_call.kwargs["top_k"] == 20  # 5 * 4 = default fetch_k


# --- Knowledge ID filtering tests ---


async def test_knowledge_id_filters_passed_to_methods():
    vector = _mock_method(
        "vector",
        [
            RetrievedChunk(chunk_id="c1", source_id="s1", content="text", score=0.9),
        ],
    )
    service = RetrievalService(retrieval_methods=[vector], top_k=5)
    await service.retrieve(query="test", knowledge_id="kb-42")
    call_kwargs = vector.search.call_args.kwargs
    assert call_kwargs["filters"] == {"knowledge_id": "kb-42"}
    assert call_kwargs["knowledge_id"] == "kb-42"


async def test_no_knowledge_id_passes_none_filters():
    vector = _mock_method(
        "vector",
        [
            RetrievedChunk(chunk_id="c1", source_id="s1", content="text", score=0.9),
        ],
    )
    service = RetrievalService(retrieval_methods=[vector], top_k=5)
    await service.retrieve(query="test")
    call_kwargs = vector.search.call_args.kwargs
    assert call_kwargs["filters"] is None
    assert call_kwargs["knowledge_id"] is None


# --- fetch_k calculation test ---


async def test_fetch_k_is_four_times_top_k():
    """Methods receive top_k * 4 candidates by default."""
    vector = _mock_method("vector", [])
    service = RetrievalService(retrieval_methods=[vector], top_k=3)
    await service.retrieve(query="test")
    assert vector.search.call_args.kwargs["top_k"] == 12  # 3 * 4


# --- Public API surface tests ---


def test_retrieval_service_exposes_public_methods_iterator() -> None:
    svc = RetrievalService(retrieval_methods=[])
    assert list(svc.methods) == []


def test_retrieval_service_methods_returns_configured_list() -> None:
    from unittest.mock import MagicMock

    from rfnry_knowledge.retrieval.base import BaseRetrievalMethod

    method = MagicMock(spec=BaseRetrievalMethod)
    svc = RetrievalService(retrieval_methods=[method])
    assert svc.methods == [method]
