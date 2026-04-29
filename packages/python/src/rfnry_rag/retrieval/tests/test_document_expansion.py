"""Document expansion at index time.

Tests pinning the public contract of ``DocumentExpansionConfig``,
``ChunkedContent.synthetic_queries`` / ``text_for_embedding`` /
``text_for_bm25``, and the ``expand_chunks`` helper.

Identifier hygiene: tests use abstract names (``chunk_a``, ``passage_1``,
``q1``) so the fixtures themselves do not seed bias-listed terms.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rfnry_rag.retrieval.common.errors import ConfigurationError, IngestionError
from rfnry_rag.retrieval.modules.ingestion.chunk.expand import expand_chunks
from rfnry_rag.retrieval.modules.ingestion.models import ChunkedContent
from rfnry_rag.retrieval.server import DocumentExpansionConfig

# ---------------------------------------------------------------------------
# DocumentExpansionConfig — bounds + invariants
# ---------------------------------------------------------------------------


def test_document_expansion_config_requires_lm_client_when_enabled() -> None:
    with pytest.raises(ConfigurationError, match="lm_client"):
        DocumentExpansionConfig(enabled=True, lm_client=None)


def test_document_expansion_config_bounds_num_queries() -> None:
    # Boundary failures
    with pytest.raises(ConfigurationError, match="num_queries"):
        DocumentExpansionConfig(num_queries=0)
    with pytest.raises(ConfigurationError, match="num_queries"):
        DocumentExpansionConfig(num_queries=21)
    # Valid values
    for n in (1, 5, 20):
        cfg = DocumentExpansionConfig(num_queries=n)
        assert cfg.num_queries == n


def test_document_expansion_config_bounds_concurrency() -> None:
    with pytest.raises(ConfigurationError, match="concurrency"):
        DocumentExpansionConfig(concurrency=0)
    with pytest.raises(ConfigurationError, match="concurrency"):
        DocumentExpansionConfig(concurrency=101)
    for n in (1, 5, 100):
        cfg = DocumentExpansionConfig(concurrency=n)
        assert cfg.concurrency == n


# ---------------------------------------------------------------------------
# ChunkedContent.embedding_text — folds in synthetic queries
# ---------------------------------------------------------------------------


def test_chunked_content_embedding_text_appends_synthetic_queries() -> None:
    chunk = ChunkedContent(
        content="passage_1",
        contextualized="ctx passage_1",
        synthetic_queries=["q1", "q2"],
    )
    text = chunk.embedding_text
    assert "ctx passage_1" in text
    assert "q1" in text
    assert "q2" in text


def test_chunked_content_embedding_text_no_change_when_queries_empty() -> None:
    chunk_with_ctx = ChunkedContent(content="passage_1", contextualized="ctx passage_1")
    assert chunk_with_ctx.embedding_text == "ctx passage_1"

    chunk_no_ctx = ChunkedContent(content="passage_1")
    assert chunk_no_ctx.embedding_text == "passage_1"


# ---------------------------------------------------------------------------
# expand_chunks — concurrency, attachment, error propagation
# ---------------------------------------------------------------------------


def _make_chunk(idx: int) -> ChunkedContent:
    return ChunkedContent(content=f"passage_{idx}", chunk_index=idx)


async def test_expand_chunks_calls_llm_per_chunk_with_concurrency_bound() -> None:
    """Each chunk gets exactly one BAML call, and at most ``concurrency`` are in flight."""
    in_flight = 0
    max_in_flight = 0
    lock = asyncio.Lock()
    call_count = 0

    async def fake_generate(passage: str, num_queries: int, baml_options: dict) -> object:
        nonlocal in_flight, max_in_flight, call_count
        async with lock:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            call_count += 1
        await asyncio.sleep(0.01)
        async with lock:
            in_flight -= 1
        return SimpleNamespace(queries=[f"q_for_{passage}"])

    cfg = DocumentExpansionConfig(
        enabled=True,
        num_queries=1,
        lm_client=MagicMock(),
        concurrency=3,
    )
    chunks = [_make_chunk(i) for i in range(10)]
    registry = MagicMock()

    with patch("rfnry_rag.retrieval.modules.ingestion.chunk.expand.b") as mock_b:
        mock_b.GenerateSyntheticQueries = AsyncMock(side_effect=fake_generate)
        await expand_chunks(chunks, cfg, registry)

    assert call_count == 10
    assert max_in_flight <= 3
    assert max_in_flight >= 2  # sanity: actually parallelised, not strictly serial


async def test_expand_chunks_attaches_queries_to_chunk_dataclass() -> None:
    cfg = DocumentExpansionConfig(
        enabled=True,
        num_queries=3,
        lm_client=MagicMock(),
        concurrency=1,
    )
    chunks = [_make_chunk(0)]
    registry = MagicMock()

    with patch("rfnry_rag.retrieval.modules.ingestion.chunk.expand.b") as mock_b:
        mock_b.GenerateSyntheticQueries = AsyncMock(
            return_value=SimpleNamespace(queries=["a", "b", "c"]),
        )
        await expand_chunks(chunks, cfg, registry)

    assert chunks[0].synthetic_queries == ["a", "b", "c"]


async def test_expand_chunks_propagates_lm_failure_as_ingestion_error() -> None:
    cfg = DocumentExpansionConfig(
        enabled=True,
        num_queries=2,
        lm_client=MagicMock(),
        concurrency=1,
    )
    chunks = [_make_chunk(7)]
    registry = MagicMock()

    with patch("rfnry_rag.retrieval.modules.ingestion.chunk.expand.b") as mock_b:
        mock_b.GenerateSyntheticQueries = AsyncMock(side_effect=RuntimeError("boom"))
        with pytest.raises(IngestionError, match="chunk_index=7"):
            await expand_chunks(chunks, cfg, registry)
