from unittest.mock import AsyncMock, patch

import pytest

from rfnry_rag.providers import LanguageModelClient, LanguageModelProvider
from rfnry_rag.retrieval.judging import RetrievalJudgment


def _make_lm_client():
    return LanguageModelClient(provider=LanguageModelProvider(provider="openai", model="gpt-4o-mini"))


class TestRetrievalJudgment:
    @pytest.fixture
    def judge(self):
        return RetrievalJudgment(
            lm_client=_make_lm_client(),
            knowledge_description="Industrial air filtration manuals",
        )

    async def test_should_retrieve_true(self, judge):
        mock_result = AsyncMock()
        mock_result.should_retrieve = True
        mock_result.confidence = 0.95
        mock_result.reasoning = "Query asks about specific filter specs"

        with patch("rfnry_rag.retrieval.judging.b.JudgeRetrievalNecessity", return_value=mock_result):
            result = await judge.should_retrieve("What is the pressure drop for FBD-20254?")

        assert result.should_retrieve is True
        assert result.confidence == 0.95

    async def test_should_retrieve_false(self, judge):
        mock_result = AsyncMock()
        mock_result.should_retrieve = False
        mock_result.confidence = 0.9
        mock_result.reasoning = "General knowledge question"

        with patch("rfnry_rag.retrieval.judging.b.JudgeRetrievalNecessity", return_value=mock_result):
            result = await judge.should_retrieve("What is photosynthesis?")

        assert result.should_retrieve is False

    async def test_defaults_to_retrieve_on_failure(self, judge):
        with patch(
            "rfnry_rag.retrieval.judging.b.JudgeRetrievalNecessity",
            side_effect=Exception("LLM error"),
        ):
            result = await judge.should_retrieve("Some query")

        assert result.should_retrieve is True
        assert result.confidence == 0.0

    async def test_clamps_confidence(self, judge):
        mock_result = AsyncMock()
        mock_result.should_retrieve = True
        mock_result.confidence = 1.5
        mock_result.reasoning = "High confidence"

        with patch("rfnry_rag.retrieval.judging.b.JudgeRetrievalNecessity", return_value=mock_result):
            result = await judge.should_retrieve("Query")

        assert result.confidence == 1.0
