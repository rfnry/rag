from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from rfnry_rag.retrieval.common.models import RetrievedChunk
from rfnry_rag.retrieval.modules.generation.models import QueryResult, StreamEvent
from rfnry_rag.retrieval.server import RagEngine, RagServerConfig


def _chunk(chunk_id: str, score: float = 0.9) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=chunk_id, source_id="s1", content="text", score=score)


def _query_result(answer: str = "The answer is 42.") -> QueryResult:
    return QueryResult(answer=answer, sources=[], grounded=True, confidence=0.9)


def _make_server() -> RagEngine:
    config = MagicMock(spec=RagServerConfig)
    config.retrieval = SimpleNamespace(history_window=3)
    server = RagEngine.__new__(RagEngine)
    server._config = config
    server._initialized = True
    server._retrieval_service = AsyncMock()
    server._retrieval_service.retrieve = AsyncMock(return_value=[_chunk("c1"), _chunk("c2")])
    server._structured_retrieval = None
    server._generation_service = AsyncMock()
    server._generation_service.generate = AsyncMock(return_value=_query_result())
    server._step_service = None
    server._knowledge_manager = None
    server._ingestion_service = None
    server._structured_ingestion = None
    server._retrieval_namespace = None
    server._ingestion_namespace = None
    server._tree_indexing_service = None
    server._tree_search_service = None
    return server


class TestServerQuery:
    async def test_query_calls_retrieval_then_generation(self):
        server = _make_server()
        result = await server.query("What is the rating?", knowledge_id="k1")

        server._retrieval_service.retrieve.assert_awaited_once()
        server._generation_service.generate.assert_awaited_once()
        assert result.answer == "The answer is 42."

    async def test_query_passes_original_text_to_generation(self):
        server = _make_server()
        await server.query("my question", history=[("prev", "answer")])

        gen_call = server._generation_service.generate.call_args
        assert gen_call.kwargs["query"] == "my question"

    async def test_query_enriches_retrieval_with_history(self):
        server = _make_server()
        await server.query("follow up", history=[("what is X?", "X is Y")])

        retrieval_call = server._retrieval_service.retrieve.call_args
        retrieval_query = retrieval_call.kwargs["query"]
        assert "follow up" in retrieval_query
        assert "what is X?" in retrieval_query

    async def test_query_without_generation_raises(self):
        server = _make_server()
        server._generation_service = None
        with pytest.raises(RuntimeError, match="query\\(\\) requires generation"):
            await server.query("test")

    async def test_query_not_initialized_raises(self):
        server = _make_server()
        server._initialized = False
        with pytest.raises(RuntimeError, match="not initialized"):
            await server.query("test")


class TestServerQueryWithStructured:
    async def test_query_merges_structured_results(self):
        server = _make_server()
        server._structured_retrieval = AsyncMock()
        server._structured_retrieval.retrieve = AsyncMock(return_value=[_chunk("s1", 0.95)])

        await server.query("test")

        gen_call = server._generation_service.generate.call_args
        chunks = gen_call.kwargs["chunks"]
        chunk_ids = {c.chunk_id for c in chunks}
        assert "c1" in chunk_ids
        assert "s1" in chunk_ids

    async def test_merge_deduplicates_by_chunk_id(self):
        server = _make_server()
        server._structured_retrieval = AsyncMock()
        server._structured_retrieval.retrieve = AsyncMock(return_value=[_chunk("c1", 0.5)])

        await server.query("test")

        gen_call = server._generation_service.generate.call_args
        chunks = gen_call.kwargs["chunks"]
        ids = [c.chunk_id for c in chunks]
        assert ids.count("c1") == 1

    async def test_merge_gives_rank1_structured_competitive_position(self):
        """With RRF, rank-1 results from both lists should end up in the top of
        the merged output — regardless of raw-score scale difference."""
        server = _make_server()
        server._structured_retrieval = AsyncMock()
        server._structured_retrieval.retrieve = AsyncMock(return_value=[_chunk("high", 0.99)])

        await server.query("test")

        gen_call = server._generation_service.generate.call_args
        chunks = gen_call.kwargs["chunks"]
        # Unstructured returned [c1, c2]; structured returned [high].
        # Under RRF, c1 and high are tied rank-1, both should lead over c2.
        top_two_ids = {chunks[0].chunk_id, chunks[1].chunk_id}
        assert "high" in top_two_ids
        assert "c1" in top_two_ids


class TestServerRetrieve:
    async def test_retrieve_returns_chunks_no_generation(self):
        server = _make_server()
        server._generation_service = None
        chunks = await server.retrieve("test")

        assert len(chunks) == 2
        assert chunks[0].chunk_id == "c1"

    async def test_retrieve_merges_structured(self):
        server = _make_server()
        server._structured_retrieval = AsyncMock()
        server._structured_retrieval.retrieve = AsyncMock(return_value=[_chunk("extra", 0.95)])

        chunks = await server.retrieve("test")
        ids = {c.chunk_id for c in chunks}
        assert "extra" in ids


class TestServerQueryStream:
    async def test_query_stream_yields_events(self):
        server = _make_server()

        async def fake_stream(query, chunks, history, system_prompt=None):
            yield StreamEvent(type="chunk", content="Hello")
            yield StreamEvent(type="sources", sources=[], grounded=True, confidence=0.9)

        server._generation_service.generate_stream = fake_stream

        events = [event async for event in server.query_stream("test")]
        assert len(events) == 2
        assert events[0].type == "chunk"
        assert events[1].type == "sources"


class TestServerGenerateStep:
    async def test_generate_step_delegates_to_step_service(self):
        server = _make_server()
        server._step_service = AsyncMock()
        server._step_service.generate_step = AsyncMock()

        await server.generate_step("query", [_chunk("c1")], context="prior reasoning")

        server._step_service.generate_step.assert_awaited_once_with(
            query="query", chunks=[_chunk("c1")], context="prior reasoning"
        )

    async def test_generate_step_without_service_raises(self):
        server = _make_server()
        with pytest.raises(RuntimeError, match="step_lm_client"):
            await server.generate_step("query", [])

    async def test_generate_step_rejects_oversize_query(self):
        server = _make_server()
        server._step_service = AsyncMock()
        with pytest.raises(ValueError, match="query exceeds"):
            await server.generate_step(query="x" * 40_000, chunks=[])


class TestMinScore:
    async def test_retrieve_filters_by_min_score(self):
        server = _make_server()
        server._retrieval_service.retrieve = AsyncMock(
            return_value=[_chunk("high", 0.9), _chunk("mid", 0.5), _chunk("low", 0.3)]
        )
        chunks = await server.retrieve("test", min_score=0.4)
        ids = [c.chunk_id for c in chunks]
        assert "high" in ids
        assert "mid" in ids
        assert "low" not in ids

    async def test_retrieve_no_filter_when_none(self):
        server = _make_server()
        server._retrieval_service.retrieve = AsyncMock(return_value=[_chunk("a", 0.9), _chunk("b", 0.1)])
        chunks = await server.retrieve("test", min_score=None)
        assert len(chunks) == 2

    async def test_query_filters_by_min_score(self):
        server = _make_server()
        server._retrieval_service.retrieve = AsyncMock(return_value=[_chunk("high", 0.9), _chunk("low", 0.2)])
        await server.query("test", min_score=0.5)

        gen_call = server._generation_service.generate.call_args
        chunks = gen_call.kwargs["chunks"]
        ids = [c.chunk_id for c in chunks]
        assert "high" in ids
        assert "low" not in ids


class TestBuildRetrievalQuery:
    def test_no_history_returns_text(self):
        server = _make_server()
        assert server._build_retrieval_query("hello", None) == "hello"

    def test_history_appends_context(self):
        server = _make_server()
        result = server._build_retrieval_query("follow up", [("what is X?", "X is Y")])
        assert "follow up" in result
        assert "what is X?" in result

    def test_uses_last_3_exchanges(self):
        server = _make_server()
        history = [(f"q{i}", f"a{i}") for i in range(5)]
        result = server._build_retrieval_query("current", history)
        assert "q2" in result
        assert "q3" in result
        assert "q4" in result
        assert "q0" not in result


class TestMergeRetrievalResults:
    """Smoke tests for `_merge_retrieval_results` contract. The scoring
    semantics (RRF) are exercised in test_fusion.py::TestMergeRetrievalResults."""

    def test_dedup_by_chunk_id(self):
        merged = RagEngine._merge_retrieval_results(
            [_chunk("a", 0.9)],
            [_chunk("a", 0.5)],
        )
        assert len(merged) == 1

    def test_returns_all_unique_chunks(self):
        merged = RagEngine._merge_retrieval_results(
            [_chunk("a", 0.04)],
            [_chunk("b", 0.9)],
        )
        ids = {c.chunk_id for c in merged}
        assert ids == {"a", "b"}
