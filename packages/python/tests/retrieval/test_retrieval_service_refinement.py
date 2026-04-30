from types import SimpleNamespace
from unittest.mock import AsyncMock

from rfnry_rag.retrieval.common.models import RetrievedChunk
from rfnry_rag.retrieval.search.service import RetrievalService


def _chunk(chunk_id: str, score: float = 0.9) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=chunk_id, source_id="s1", content="text", score=score)


def _make_service(chunk_refiner=None, reranking=None):
    mock_vector = SimpleNamespace(
        name="vector",
        weight=1.0,
        top_k=None,
        search=AsyncMock(return_value=[_chunk("v1", 0.9), _chunk("v2", 0.8)]),
    )
    return RetrievalService(
        retrieval_methods=[mock_vector],
        top_k=5,
        chunk_refiner=chunk_refiner,
        reranking=reranking,
    )


class TestChunkRefinementWiring:
    async def test_refiner_called_after_reranking(self):
        refiner = AsyncMock()
        refiner.refine = AsyncMock(return_value=[_chunk("refined")])
        service = _make_service(chunk_refiner=refiner)

        results, _ = await service.retrieve("test query")

        refiner.refine.assert_awaited_once()
        assert results[0].chunk_id == "refined"

    async def test_refiner_receives_original_query(self):
        refiner = AsyncMock()
        refiner.refine = AsyncMock(return_value=[_chunk("r1")])
        service = _make_service(chunk_refiner=refiner)

        await service.retrieve("my question")

        call_args = refiner.refine.call_args
        assert call_args.args[0] == "my question"

    async def test_refiner_receives_reranked_results(self):
        reranker = AsyncMock()
        reranked = [_chunk("reranked1", 0.95)]
        reranker.rerank = AsyncMock(return_value=reranked)

        refiner = AsyncMock()
        refiner.refine = AsyncMock(return_value=reranked)
        service = _make_service(chunk_refiner=refiner, reranking=reranker)

        await service.retrieve("test query")

        refiner_chunks = refiner.refine.call_args.args[1]
        assert refiner_chunks[0].chunk_id == "reranked1"

    async def test_no_refiner_skips_refinement(self):
        service = _make_service(chunk_refiner=None)
        results, _ = await service.retrieve("test query")
        assert len(results) == 2
        assert results[0].chunk_id == "v1"

    async def test_refiner_skipped_on_empty_results(self):
        refiner = AsyncMock()
        refiner.refine = AsyncMock()
        mock_vector = SimpleNamespace(
            name="vector",
            weight=1.0,
            top_k=None,
            search=AsyncMock(return_value=[]),
        )

        service = RetrievalService(retrieval_methods=[mock_vector], top_k=5, chunk_refiner=refiner)
        results, _ = await service.retrieve("test query")

        refiner.refine.assert_not_awaited()
        assert results == []
