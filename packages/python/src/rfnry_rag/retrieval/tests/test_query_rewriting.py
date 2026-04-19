from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from baml_py import errors as baml_errors

from rfnry_rag.retrieval.common.models import RetrievedChunk
from rfnry_rag.retrieval.modules.retrieval.search.rewriting.hyde import HyDeRewriting
from rfnry_rag.retrieval.modules.retrieval.search.rewriting.multi_query import MultiQueryRewriting
from rfnry_rag.retrieval.modules.retrieval.search.rewriting.step_back import StepBackRewriting
from rfnry_rag.retrieval.modules.retrieval.search.service import RetrievalService
from rfnry_rag.retrieval.server import RetrievalConfig


async def test_hyde_rewriter_returns_hypothetical_passage():
    """HyDE should return a single hypothetical passage from BAML."""
    mock_lm_client = MagicMock()
    rewriter = HyDeRewriting(lm_client=mock_lm_client)

    mock_result = MagicMock()
    mock_result.passage = "The 20x25x4 MERV 13 filter has an initial pressure drop of 0.25 inches WG."

    with (
        patch("rfnry_rag.retrieval.modules.retrieval.search.rewriting.hyde.b") as mock_b,
        patch("rfnry_rag.retrieval.modules.retrieval.search.rewriting.hyde.build_registry") as mock_registry,
    ):
        mock_b.GenerateHypotheticalDocument = AsyncMock(return_value=mock_result)
        mock_registry.return_value = MagicMock()

        result = await rewriter.rewrite("What is the pressure drop for a 20x25x4 filter?")

    assert isinstance(result, list)
    assert len(result) == 1
    assert "pressure drop" in result[0]


async def test_hyde_rewriter_with_conversation_context():
    """HyDE should include conversation context in the BAML call."""
    mock_lm_client = MagicMock()
    rewriter = HyDeRewriting(lm_client=mock_lm_client)

    mock_result = MagicMock()
    mock_result.passage = "Hypothetical answer about filters."

    with (
        patch("rfnry_rag.retrieval.modules.retrieval.search.rewriting.hyde.b") as mock_b,
        patch("rfnry_rag.retrieval.modules.retrieval.search.rewriting.hyde.build_registry") as mock_registry,
    ):
        mock_b.GenerateHypotheticalDocument = AsyncMock(return_value=mock_result)
        mock_registry.return_value = MagicMock()

        result = await rewriter.rewrite(
            "What about the 20x25x4?",
            conversation_context="What filters do you carry?",
        )

    assert len(result) == 1
    call_args = mock_b.GenerateHypotheticalDocument.call_args
    assert "Conversation context" in call_args[0][0]


async def test_multi_query_rewriter_returns_variants():
    """Multi-query should return the configured number of variants."""
    mock_lm_client = MagicMock()
    rewriter = MultiQueryRewriting(lm_client=mock_lm_client, num_variants=3)

    mock_result = MagicMock()
    mock_result.variants = [
        "How to install an air filter in an HVAC system",
        "Filter replacement steps and instructions",
        "Air filter mounting and securing procedure",
    ]

    with (
        patch("rfnry_rag.retrieval.modules.retrieval.search.rewriting.multi_query.b") as mock_b,
        patch("rfnry_rag.retrieval.modules.retrieval.search.rewriting.multi_query.build_registry") as mock_registry,
    ):
        mock_b.GenerateQueryVariants = AsyncMock(return_value=mock_result)
        mock_registry.return_value = MagicMock()

        result = await rewriter.rewrite("Filter installation")

    assert isinstance(result, list)
    assert len(result) == 3


async def test_multi_query_default_num_variants():
    """Multi-query should default to 3 variants."""
    mock_lm_client = MagicMock()
    rewriter = MultiQueryRewriting(lm_client=mock_lm_client)
    assert rewriter.num_variants == 3


async def test_multi_query_passes_num_variants_to_baml():
    """Multi-query should pass num_variants to the BAML call."""
    mock_lm_client = MagicMock()
    rewriter = MultiQueryRewriting(lm_client=mock_lm_client, num_variants=2)

    mock_result = MagicMock()
    mock_result.variants = ["variant 1", "variant 2"]

    with (
        patch("rfnry_rag.retrieval.modules.retrieval.search.rewriting.multi_query.b") as mock_b,
        patch("rfnry_rag.retrieval.modules.retrieval.search.rewriting.multi_query.build_registry") as mock_registry,
    ):
        mock_b.GenerateQueryVariants = AsyncMock(return_value=mock_result)
        mock_registry.return_value = MagicMock()

        await rewriter.rewrite("test query")

    call_args = mock_b.GenerateQueryVariants.call_args
    assert call_args[0][1] == 2


async def test_step_back_rewriter_returns_broader_query():
    """Step-back should return a single broader query."""
    mock_lm_client = MagicMock()
    rewriter = StepBackRewriting(lm_client=mock_lm_client)

    mock_result = MagicMock()
    mock_result.broader_query = "MERV 13 filter specifications and performance ratings"

    with (
        patch("rfnry_rag.retrieval.modules.retrieval.search.rewriting.step_back.b") as mock_b,
        patch("rfnry_rag.retrieval.modules.retrieval.search.rewriting.step_back.build_registry") as mock_registry,
    ):
        mock_b.GenerateStepBackQuery = AsyncMock(return_value=mock_result)
        mock_registry.return_value = MagicMock()

        result = await rewriter.rewrite("What is the MERV 13 rating for a 20x25x4 filter at 500 FPM?")

    assert isinstance(result, list)
    assert len(result) == 1
    assert "specifications" in result[0]


async def test_step_back_rewriter_with_context():
    """Step-back should include conversation context in the BAML call."""
    mock_lm_client = MagicMock()
    rewriter = StepBackRewriting(lm_client=mock_lm_client)

    mock_result = MagicMock()
    mock_result.broader_query = "Air filter performance overview"

    with (
        patch("rfnry_rag.retrieval.modules.retrieval.search.rewriting.step_back.b") as mock_b,
        patch("rfnry_rag.retrieval.modules.retrieval.search.rewriting.step_back.build_registry") as mock_registry,
    ):
        mock_b.GenerateStepBackQuery = AsyncMock(return_value=mock_result)
        mock_registry.return_value = MagicMock()

        result = await rewriter.rewrite(
            "What about at 500 FPM?",
            conversation_context="What are the specs for the 20x25x4?",
        )

    assert len(result) == 1
    call_args = mock_b.GenerateStepBackQuery.call_args
    assert "Conversation context" in call_args[0][0]


async def test_hyde_returns_empty_on_baml_validation_error():
    """HyDE should return [] when BAML fails to parse the LLM response."""
    mock_lm_client = MagicMock()
    rewriter = HyDeRewriting(lm_client=mock_lm_client)

    with (
        patch("rfnry_rag.retrieval.modules.retrieval.search.rewriting.hyde.b") as mock_b,
        patch("rfnry_rag.retrieval.modules.retrieval.search.rewriting.hyde.build_registry") as mock_registry,
    ):
        mock_b.GenerateHypotheticalDocument = AsyncMock(
            side_effect=baml_errors.BamlValidationError(
                prompt="p", message="parse error", raw_output="bad", detailed_message="detail"
            )
        )
        mock_registry.return_value = MagicMock()

        result = await rewriter.rewrite("test query")

    assert result == []


async def test_multi_query_returns_empty_on_exception():
    """Multi-query should return [] on any exception."""
    mock_lm_client = MagicMock()
    rewriter = MultiQueryRewriting(lm_client=mock_lm_client)

    with (
        patch("rfnry_rag.retrieval.modules.retrieval.search.rewriting.multi_query.b") as mock_b,
        patch("rfnry_rag.retrieval.modules.retrieval.search.rewriting.multi_query.build_registry") as mock_registry,
    ):
        mock_b.GenerateQueryVariants = AsyncMock(side_effect=RuntimeError("LLM timeout"))
        mock_registry.return_value = MagicMock()

        result = await rewriter.rewrite("test query")

    assert result == []


async def test_step_back_returns_empty_on_exception():
    """Step-back should return [] on any exception."""
    mock_lm_client = MagicMock()
    rewriter = StepBackRewriting(lm_client=mock_lm_client)

    with (
        patch("rfnry_rag.retrieval.modules.retrieval.search.rewriting.step_back.b") as mock_b,
        patch("rfnry_rag.retrieval.modules.retrieval.search.rewriting.step_back.build_registry") as mock_registry,
    ):
        mock_b.GenerateStepBackQuery = AsyncMock(side_effect=RuntimeError("Network error"))
        mock_registry.return_value = MagicMock()

        result = await rewriter.rewrite("test query")

    assert result == []


def _make_retrieval_service(query_rewriter=None):
    """Helper to build a RetrievalService with mocked search backends."""
    mock_vector = SimpleNamespace(
        name="vector",
        weight=1.0,
        top_k=None,
        search=AsyncMock(
            return_value=[
                RetrievedChunk(chunk_id="chunk-1", source_id="src-1", content="Vector result", score=0.8),
            ]
        ),
    )
    return RetrievalService(
        retrieval_methods=[mock_vector],
        reranking=None,
        top_k=5,
        query_rewriter=query_rewriter,
    )


async def test_retrieve_without_rewriter_unchanged():
    """When no rewriter is configured, retrieve() works exactly as before."""
    service = _make_retrieval_service(query_rewriter=None)
    results = await service.retrieve(query="test query", knowledge_id="kb-1")
    assert len(results) == 1
    assert results[0].chunk_id == "chunk-1"


async def test_retrieve_with_rewriter_searches_original_and_rewritten():
    """When a rewriter is configured, retrieve() searches all queries."""
    mock_rewriter = AsyncMock()
    mock_rewriter.rewrite = AsyncMock(return_value=["rewritten query 1", "rewritten query 2"])

    mock_vector = SimpleNamespace(
        name="vector",
        weight=1.0,
        top_k=None,
        search=AsyncMock(
            side_effect=[
                [RetrievedChunk(chunk_id="orig-1", source_id="s1", content="Original result", score=0.9)],
                [RetrievedChunk(chunk_id="rw-1", source_id="s2", content="Rewritten result 1", score=0.85)],
                [RetrievedChunk(chunk_id="rw-2", source_id="s3", content="Rewritten result 2", score=0.7)],
            ]
        ),
    )

    service = RetrievalService(
        retrieval_methods=[mock_vector],
        reranking=None,
        top_k=5,
        query_rewriter=mock_rewriter,
    )
    results = await service.retrieve(query="test query", knowledge_id="kb-1")

    assert mock_vector.search.call_count == 3
    chunk_ids = {r.chunk_id for r in results}
    assert "orig-1" in chunk_ids
    assert "rw-1" in chunk_ids
    assert "rw-2" in chunk_ids


async def test_retrieve_with_rewriter_deduplicates_via_fusion():
    """When multiple queries return the same chunk, RRF deduplicates and boosts."""
    mock_rewriter = AsyncMock()
    mock_rewriter.rewrite = AsyncMock(return_value=["rewritten query"])

    shared_chunk = RetrievedChunk(chunk_id="shared-1", source_id="s1", content="Shared", score=0.8)

    mock_vector = SimpleNamespace(
        name="vector",
        weight=1.0,
        top_k=None,
        search=AsyncMock(
            side_effect=[
                [shared_chunk],
                [shared_chunk],
            ]
        ),
    )

    service = RetrievalService(
        retrieval_methods=[mock_vector],
        reranking=None,
        top_k=5,
        query_rewriter=mock_rewriter,
    )
    results = await service.retrieve(query="test query")

    assert len(results) == 1
    assert results[0].chunk_id == "shared-1"


async def test_retrieve_with_rewriter_failure_still_works():
    """If the rewriter raises an exception, retrieve() still works with the original query."""
    mock_rewriter = AsyncMock()
    mock_rewriter.rewrite = AsyncMock(side_effect=RuntimeError("LLM down"))

    service = _make_retrieval_service(query_rewriter=mock_rewriter)
    results = await service.retrieve(query="test query", knowledge_id="kb-1")

    assert len(results) == 1
    assert results[0].chunk_id == "chunk-1"


async def test_reranker_receives_original_query_not_rewritten():
    """The reranker should always score against the original user query."""
    mock_rewriter = AsyncMock()
    mock_rewriter.rewrite = AsyncMock(return_value=["rewritten version"])

    mock_reranker = AsyncMock()
    mock_reranker.rerank = AsyncMock(
        return_value=[RetrievedChunk(chunk_id="c1", source_id="s1", content="Result", score=0.95)]
    )

    mock_vector = SimpleNamespace(
        name="vector",
        weight=1.0,
        top_k=None,
        search=AsyncMock(
            return_value=[
                RetrievedChunk(chunk_id="c1", source_id="s1", content="Result", score=0.8),
            ]
        ),
    )

    service = RetrievalService(
        retrieval_methods=[mock_vector],
        reranking=mock_reranker,
        top_k=5,
        query_rewriter=mock_rewriter,
    )
    await service.retrieve(query="original user question", knowledge_id="kb-1")

    rerank_call = mock_reranker.rerank.call_args
    assert rerank_call[0][0] == "original user question"


async def test_retrieval_config_accepts_query_rewriter():
    """RetrievalConfig should accept a query_rewriter field."""
    mock_rewriter = MagicMock()
    config = RetrievalConfig(query_rewriter=mock_rewriter)
    assert config.query_rewriter is mock_rewriter


async def test_retrieval_config_defaults_to_no_rewriter():
    """RetrievalConfig should default query_rewriter to None."""
    config = RetrievalConfig()
    assert config.query_rewriter is None
