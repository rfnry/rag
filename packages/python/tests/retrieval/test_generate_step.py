from unittest.mock import AsyncMock, patch

import pytest

from rfnry_rag.common.language_model import LanguageModelClient, LanguageModelProvider
from rfnry_rag.exceptions import GenerationError
from rfnry_rag.generation.step import StepGenerationService
from rfnry_rag.retrieval.common.models import RetrievedChunk


def _make_chunk(content: str) -> RetrievedChunk:
    return RetrievedChunk(chunk_id="c1", source_id="s1", content=content, score=0.9, source_metadata={"name": "Test"})


def _make_lm_client():
    return LanguageModelClient(provider=LanguageModelProvider(provider="openai", model="gpt-4o-mini"))


class TestStepGenerationService:
    @pytest.fixture
    def service(self):
        return StepGenerationService(lm_client=_make_lm_client())

    async def test_generates_intermediate_step(self, service):
        mock_result = AsyncMock()
        mock_result.text = "The manual mentions SAE 30 oil but does not specify quantity."
        mock_result.is_final = False

        with patch("rfnry_rag.generation.step.b.GenerateReasoningStep", return_value=mock_result):
            result = await service.generate_step(
                query="How much oil does it need?",
                chunks=[_make_chunk("Use SAE 30 oil for this model.")],
            )

        assert result.done is False
        assert "SAE 30" in result.text

    async def test_generates_final_answer(self, service):
        mock_result = AsyncMock()
        mock_result.text = "The engine requires 2.5 quarts of SAE 30 oil."
        mock_result.is_final = True

        with patch("rfnry_rag.generation.step.b.GenerateReasoningStep", return_value=mock_result):
            result = await service.generate_step(
                query="How much oil does it need?",
                chunks=[_make_chunk("Oil capacity: 2.5 quarts SAE 30.")],
                context="The manual mentions SAE 30 oil.",
            )

        assert result.done is True
        assert "2.5 quarts" in result.text

    async def test_empty_query_raises(self, service):
        with pytest.raises(GenerationError):
            await service.generate_step(query="", chunks=[])

    async def test_baml_failure_raises_generation_error(self, service):
        with (
            patch(
                "rfnry_rag.generation.step.b.GenerateReasoningStep",
                side_effect=Exception("LLM error"),
            ),
            pytest.raises(GenerationError, match="GenerateReasoningStep failed"),
        ):
            await service.generate_step(query="question", chunks=[_make_chunk("content")])

    async def test_empty_chunks_uses_placeholder(self, service):
        mock_result = AsyncMock()
        mock_result.text = "No relevant context was found."
        mock_result.is_final = True

        with patch("rfnry_rag.generation.step.b.GenerateReasoningStep", return_value=mock_result) as mock_call:
            await service.generate_step(query="question", chunks=[])

        call_args = mock_call.call_args
        context_arg = call_args.kwargs.get("context", "")
        assert "(No context retrieved)" in context_arg
