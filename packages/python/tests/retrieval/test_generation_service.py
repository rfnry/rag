from unittest.mock import AsyncMock, patch

import pytest

from rfnry_rag.exceptions import GenerationError
from rfnry_rag.generation.grounding import DEFAULT_ESCALATION
from rfnry_rag.generation.models import RelevanceResult
from rfnry_rag.generation.service import GenerationService
from rfnry_rag.models import RetrievedChunk
from rfnry_rag.providers import LanguageModel, LanguageModelClient


def _chunk(chunk_id: str = "c1", score: float = 0.9) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        source_id="s1",
        content="The filter has a MERV 13 rating.",
        score=score,
        page_number=1,
        source_metadata={"name": "Manual"},
    )


def _lm_client() -> LanguageModelClient:
    return LanguageModelClient(lm=LanguageModel(provider="openai", model="gpt-4o-mini"))


def _make_service(
    grounding_enabled: bool = False,
    grounding_threshold: float = 0.5,
    relevance_gate_enabled: bool = False,
    guiding_enabled: bool = False,
) -> GenerationService:
    lm_client = _lm_client()
    relevance_lm = lm_client if relevance_gate_enabled else None

    return GenerationService(
        lm_client=lm_client,
        system_prompt="You are helpful.",
        grounding_enabled=grounding_enabled,
        grounding_threshold=grounding_threshold,
        relevance_gate_enabled=relevance_gate_enabled,
        guiding_enabled=guiding_enabled,
        relevance_gate_lm_client=relevance_lm,
    )


def _patch_generate(answer="The MERV 13 filter is rated for fine particles."):
    return patch.object(LanguageModelClient, "generate_text", new_callable=AsyncMock, return_value=answer)


class _FakeStream:
    def __init__(self, deltas: list[str]) -> None:
        self._deltas = deltas

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        for delta in self._deltas:
            yield delta


def _patch_stream():
    return patch.object(
        LanguageModelClient,
        "generate_text_stream",
        return_value=_FakeStream(["The ", "answer ", "is ", "42."]),
    )


class TestGenerationServiceGenerate:
    async def test_generate_returns_answer(self):
        service = _make_service()
        with _patch_generate():
            result = await service.generate("What is the rating?", [_chunk()])
        assert result.answer is not None
        assert result.grounded is True
        assert len(result.sources) == 1

    async def test_generate_empty_query_raises(self):
        service = _make_service()
        with pytest.raises(GenerationError):
            await service.generate("", [_chunk()])

    async def test_generate_builds_sources_from_chunks(self):
        service = _make_service()
        with _patch_generate():
            result = await service.generate("query", [_chunk("c1", 0.9), _chunk("c2", 0.8)])
        assert len(result.sources) == 2
        assert result.sources[0].source_id == "s1"
        assert result.sources[0].name == "Manual"
        assert result.sources[0].page_number == 1


class TestScoreGate:
    async def test_rejects_low_scores(self):
        service = _make_service(grounding_enabled=True, grounding_threshold=0.8)
        with _patch_generate():
            result = await service.generate("query", [_chunk(score=0.3)])
        assert result.grounded is False
        assert result.answer == DEFAULT_ESCALATION

    async def test_passes_high_scores(self):
        service = _make_service(grounding_enabled=True, grounding_threshold=0.5)
        with _patch_generate():
            result = await service.generate("query", [_chunk(score=0.9)])
        assert result.grounded is True

    async def test_rejects_empty_chunks(self):
        service = _make_service(grounding_enabled=True, grounding_threshold=0.5)
        with _patch_generate():
            result = await service.generate("query", [])
        assert result.grounded is False

    async def test_confidence_from_max_score(self):
        service = _make_service(grounding_enabled=True, grounding_threshold=0.3)
        with _patch_generate():
            result = await service.generate("query", [_chunk(score=0.75)])
        assert result.confidence == 0.75


class TestRelevanceGate:
    async def test_rejection(self):
        service = _make_service(grounding_enabled=True, relevance_gate_enabled=True)
        relevance = RelevanceResult(answerable=False, confidence=0.2, relevant_indices=[])
        service._relevance_gate.check = AsyncMock(return_value=(False, DEFAULT_ESCALATION, relevance))

        with _patch_generate():
            result = await service.generate("query", [_chunk()])
        assert result.grounded is False

    async def test_passes(self):
        service = _make_service(grounding_enabled=True, relevance_gate_enabled=True)
        relevance = RelevanceResult(answerable=True, confidence=0.95, relevant_indices=[0])
        service._relevance_gate.check = AsyncMock(return_value=(True, None, relevance))

        with _patch_generate():
            result = await service.generate("query", [_chunk()])
        assert result.grounded is True
        assert result.confidence == 0.95

    async def test_guiding_returns_clarification(self):
        service = _make_service(grounding_enabled=True, relevance_gate_enabled=True, guiding_enabled=True)
        relevance = RelevanceResult(
            answerable=False,
            confidence=0.3,
            relevant_indices=[],
            needs_clarification=True,
            clarifying_question="Which filter type?",
            clarifying_options=["MERV 13", "HEPA"],
        )
        service._relevance_gate.check = AsyncMock(return_value=(False, None, relevance))

        with _patch_generate():
            result = await service.generate("query", [_chunk()])
        assert result.answer is None
        assert result.clarification is not None
        assert result.clarification.question == "Which filter type?"
        assert "Something else" in result.clarification.options


class TestGenerationServiceStream:
    async def test_yields_chunks_then_sources(self):
        service = _make_service()
        with _patch_stream():
            events = [event async for event in service.generate_stream("query", [_chunk()])]

        chunk_events = [e for e in events if e.type == "chunk"]
        source_events = [e for e in events if e.type == "sources"]

        assert len(chunk_events) == 4
        assert len(source_events) == 1
        assert source_events[0].grounded is True

    async def test_score_gate_rejection_yields_done(self):
        service = _make_service(grounding_enabled=True, grounding_threshold=0.9)
        with _patch_stream():
            events = [event async for event in service.generate_stream("query", [_chunk(score=0.1)])]

        assert len(events) == 1
        assert events[0].type == "done"
        assert events[0].grounded is False
