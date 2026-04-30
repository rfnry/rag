from unittest.mock import AsyncMock, patch

import pytest

from rfnry_rag.models import RetrievedChunk
from rfnry_rag.providers import LanguageModelClient, LanguageModelProvider
from rfnry_rag.retrieval.refinement.abstractive import AbstractiveRefinement
from rfnry_rag.retrieval.refinement.extractive import ExtractiveRefinement


def _make_chunk(content: str, chunk_id: str = "c1", page_number: int | None = 1) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        source_id="s1",
        content=content,
        score=0.9,
        page_number=page_number,
        source_metadata={"name": "Test Doc"},
    )


def _make_lm_client():
    return LanguageModelClient(provider=LanguageModelProvider(provider="openai", model="gpt-4o-mini"))


class TestExtractiveRefinement:
    @pytest.fixture
    def mock_embeddings(self):
        embeddings = AsyncMock()
        embeddings.model = "test-model"
        return embeddings

    async def test_empty_chunks(self, mock_embeddings):
        refiner = ExtractiveRefinement(embeddings=mock_embeddings)
        result = await refiner.refine("query", [])
        assert result == []

    async def test_preserves_metadata(self, mock_embeddings):
        """Refined chunks should preserve original metadata."""
        chunk = _make_chunk("This is a relevant sentence. This is irrelevant.", page_number=5)

        mock_embeddings.embed = AsyncMock(
            return_value=[
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        refiner = ExtractiveRefinement(embeddings=mock_embeddings, max_sentences=1)
        result = await refiner.refine("query", [chunk])

        assert len(result) == 1
        assert result[0].page_number == 5
        assert result[0].source_id == "s1"
        assert result[0].source_metadata == {"name": "Test Doc"}


class TestAbstractiveRefinement:
    async def test_empty_chunks(self):
        refiner = AbstractiveRefinement(lm_client=_make_lm_client())
        result = await refiner.refine("query", [])
        assert result == []

    async def test_compresses_chunks(self):
        chunks = [
            _make_chunk("The pressure drop is 0.5 inches WC.", "c1"),
            _make_chunk("The filter is MERV 13 rated.", "c2"),
        ]

        mock_result = AsyncMock()
        mock_result.compressed_text = "Pressure drop: 0.5 inches WC. MERV 13 rated."

        with patch(
            "rfnry_rag.retrieval.refinement.abstractive.b.CompressRetrievedContext",
            return_value=mock_result,
        ):
            refiner = AbstractiveRefinement(lm_client=_make_lm_client())
            result = await refiner.refine("What is the pressure drop?", chunks)

        assert len(result) == 1
        assert "0.5 inches" in result[0].content
        assert result[0].source_id == "s1"
        assert result[0].chunk_id == "c1"

    async def test_falls_back_on_failure(self):
        chunks = [_make_chunk("Original content.")]

        with patch(
            "rfnry_rag.retrieval.refinement.abstractive.b.CompressRetrievedContext",
            side_effect=Exception("LLM error"),
        ):
            refiner = AbstractiveRefinement(lm_client=_make_lm_client())
            result = await refiner.refine("query", chunks)

        assert len(result) == 1
        assert result[0].content == "Original content."

    async def test_falls_back_on_empty_response(self):
        chunks = [_make_chunk("Original content.")]

        mock_result = AsyncMock()
        mock_result.compressed_text = ""

        with patch(
            "rfnry_rag.retrieval.refinement.abstractive.b.CompressRetrievedContext",
            return_value=mock_result,
        ):
            refiner = AbstractiveRefinement(lm_client=_make_lm_client())
            result = await refiner.refine("query", chunks)

        assert len(result) == 1
        assert result[0].content == "Original content."
