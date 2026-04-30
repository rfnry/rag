"""Tests that each embedding provider makes a single API call per embed() invocation,
and that embed_batched() at the helper layer owns chunking and concurrency."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from rfnry_rag.ingestion.embeddings.cohere import (
    _COHERE_MAX_BATCH,
    _CohereEmbeddings,
)
from rfnry_rag.ingestion.embeddings.openai import (
    _OPENAI_MAX_BATCH,
    _OpenAIEmbeddings,
)
from rfnry_rag.ingestion.embeddings.voyage import (
    _VOYAGE_MAX_BATCH,
    _VoyageEmbeddings,
)
from rfnry_rag.providers import LanguageModelProvider

# ---------------------------------------------------------------------------
# OpenAI — single-call provider behaviour
# ---------------------------------------------------------------------------


async def test_openai_embeddings_single_call() -> None:
    """Provider embed() issues a single API call; chunking is owned by embed_batched()."""
    calls: list[int] = []

    async def fake_create(input: list[str], **kwargs: object) -> object:
        n = len(input)
        calls.append(n)
        return SimpleNamespace(data=[SimpleNamespace(embedding=[0.0]) for _ in range(n)])

    with patch("rfnry_rag.ingestion.embeddings.openai.AsyncOpenAI") as mock_cls:
        fake_client = MagicMock()
        fake_client.embeddings.create = AsyncMock(side_effect=fake_create)
        mock_cls.return_value = fake_client

        provider = LanguageModelProvider(provider="openai", model="text-embedding-3-small", api_key="sk-test")
        emb = _OpenAIEmbeddings(provider=provider)
        result = await emb.embed(["x"] * _OPENAI_MAX_BATCH)

    assert calls == [2048], f"expected single call of 2048, got {calls}"
    assert len(result) == _OPENAI_MAX_BATCH


async def test_openai_embeddings_within_limit_single_call() -> None:
    """Inputs at or below the cap must be sent in a single API call."""
    calls: list[int] = []

    async def fake_create(input: list[str], **kwargs: object) -> object:
        n = len(input)
        calls.append(n)
        return SimpleNamespace(data=[SimpleNamespace(embedding=[0.0]) for _ in range(n)])

    with patch("rfnry_rag.ingestion.embeddings.openai.AsyncOpenAI") as mock_cls:
        fake_client = MagicMock()
        fake_client.embeddings.create = AsyncMock(side_effect=fake_create)
        mock_cls.return_value = fake_client

        provider = LanguageModelProvider(provider="openai", model="text-embedding-3-small", api_key="sk-test")
        emb = _OpenAIEmbeddings(provider=provider)
        result = await emb.embed(["x"] * _OPENAI_MAX_BATCH)

    assert calls == [2048], f"expected single call of 2048, got {calls}"
    assert len(result) == _OPENAI_MAX_BATCH


# ---------------------------------------------------------------------------
# Voyage — single-call provider behaviour
# ---------------------------------------------------------------------------


async def test_voyage_embeddings_single_call() -> None:
    """Provider embed() issues a single API call; chunking is owned by embed_batched()."""
    calls: list[int] = []

    async def fake_embed(texts: list[str], **kwargs: object) -> object:
        calls.append(len(texts))
        return SimpleNamespace(embeddings=[[0.0] for _ in texts])

    with patch("rfnry_rag.ingestion.embeddings.voyage.voyageai") as mock_voyageai:
        fake_client = MagicMock()
        fake_client.embed = AsyncMock(side_effect=fake_embed)
        mock_voyageai.AsyncClient.return_value = fake_client

        provider = LanguageModelProvider(provider="voyage", model="voyage-3", api_key="vo-test")
        emb = _VoyageEmbeddings(provider=provider)
        result = await emb.embed(["x"] * _VOYAGE_MAX_BATCH)

    assert calls == [128], f"expected single call of 128, got {calls}"
    assert len(result) == _VOYAGE_MAX_BATCH


async def test_voyage_embeddings_within_limit_single_call() -> None:
    """Inputs at or below the cap must be sent in a single API call."""
    calls: list[int] = []

    async def fake_embed(texts: list[str], **kwargs: object) -> object:
        calls.append(len(texts))
        return SimpleNamespace(embeddings=[[0.0] for _ in texts])

    with patch("rfnry_rag.ingestion.embeddings.voyage.voyageai") as mock_voyageai:
        fake_client = MagicMock()
        fake_client.embed = AsyncMock(side_effect=fake_embed)
        mock_voyageai.AsyncClient.return_value = fake_client

        provider = LanguageModelProvider(provider="voyage", model="voyage-3", api_key="vo-test")
        emb = _VoyageEmbeddings(provider=provider)
        result = await emb.embed(["x"] * _VOYAGE_MAX_BATCH)

    assert calls == [128], f"expected single call of 128, got {calls}"
    assert len(result) == _VOYAGE_MAX_BATCH


# ---------------------------------------------------------------------------
# Cohere — single-call provider behaviour
# ---------------------------------------------------------------------------


async def test_cohere_embeddings_single_call() -> None:
    """Provider embed() issues a single API call; chunking is owned by embed_batched()."""
    calls: list[int] = []

    async def fake_embed(texts: list[str], **kwargs: object) -> object:
        n = len(texts)
        calls.append(n)
        embeddings_ns = SimpleNamespace(float_=[[0.0] for _ in range(n)])
        return SimpleNamespace(embeddings=embeddings_ns)

    with patch("rfnry_rag.ingestion.embeddings.cohere.cohere") as mock_cohere:
        fake_client = MagicMock()
        fake_client.embed = AsyncMock(side_effect=fake_embed)
        mock_cohere.AsyncClientV2.return_value = fake_client

        provider = LanguageModelProvider(provider="cohere", model="embed-english-v3.0", api_key="co-test")
        emb = _CohereEmbeddings(provider=provider)
        result = await emb.embed(["x"] * _COHERE_MAX_BATCH)

    assert calls == [96], f"expected single call of 96, got {calls}"
    assert len(result) == _COHERE_MAX_BATCH


async def test_cohere_embeddings_within_limit_single_call() -> None:
    """Inputs at or below the cap must be sent in a single API call."""
    calls: list[int] = []

    async def fake_embed(texts: list[str], **kwargs: object) -> object:
        n = len(texts)
        calls.append(n)
        embeddings_ns = SimpleNamespace(float_=[[0.0] for _ in range(n)])
        return SimpleNamespace(embeddings=embeddings_ns)

    with patch("rfnry_rag.ingestion.embeddings.cohere.cohere") as mock_cohere:
        fake_client = MagicMock()
        fake_client.embed = AsyncMock(side_effect=fake_embed)
        mock_cohere.AsyncClientV2.return_value = fake_client

        provider = LanguageModelProvider(provider="cohere", model="embed-english-v3.0", api_key="co-test")
        emb = _CohereEmbeddings(provider=provider)
        result = await emb.embed(["x"] * _COHERE_MAX_BATCH)

    assert calls == [96], f"expected single call of 96, got {calls}"
    assert len(result) == _COHERE_MAX_BATCH


# ---------------------------------------------------------------------------
# embed_batched helper — owns chunking and concurrency
# ---------------------------------------------------------------------------


async def test_embed_batched_chunks_large_input() -> None:
    """embed_batched splits texts into sub-batches and returns all vectors in order."""
    calls: list[int] = []

    class Fake:
        model = "fake"

        async def embed(self, texts: list[str]) -> list[list[float]]:
            calls.append(len(texts))
            return [[float(i)] for i in range(len(texts))]

        async def embedding_dimension(self) -> int:
            return 1

    from rfnry_rag.ingestion.embeddings.batching import embed_batched

    result = await embed_batched(Fake(), ["x"] * 500, batch_size=100)
    assert len(result) == 500
    # 500 texts / 100 batch_size = 5 sub-batches
    assert len(calls) == 5
    assert all(c == 100 for c in calls)


async def test_embed_batched_overlaps_sub_batches() -> None:
    """embed_batched must gather sub-batches concurrently, not await serially."""
    concurrent = 0
    max_concurrent = 0

    class Fake:
        model = "fake"

        async def embed(self, texts: list[str]) -> list[list[float]]:
            nonlocal concurrent, max_concurrent
            concurrent += 1
            max_concurrent = max(max_concurrent, concurrent)
            await asyncio.sleep(0.02)
            concurrent -= 1
            return [[0.0]] * len(texts)

        async def embedding_dimension(self) -> int:
            return 1

    from rfnry_rag.ingestion.embeddings.batching import embed_batched

    result = await embed_batched(Fake(), ["x"] * 500, batch_size=100)
    assert len(result) == 500
    assert max_concurrent >= 2, f"expected concurrent calls >= 2, got {max_concurrent}"


async def test_embed_batched_empty_input() -> None:
    """Empty input returns empty list without calling the provider."""
    calls = 0

    class Fake:
        model = "fake"

        async def embed(self, texts: list[str]) -> list[list[float]]:
            nonlocal calls
            calls += 1
            return []

        async def embedding_dimension(self) -> int:
            return 1

    from rfnry_rag.ingestion.embeddings.batching import embed_batched

    result = await embed_batched(Fake(), [], batch_size=100)
    assert result == []
    assert calls == 0


async def test_embed_batched_within_batch_size_single_call() -> None:
    """If len(texts) <= batch_size, a single embed() call is made."""
    calls = 0

    class Fake:
        model = "fake"

        async def embed(self, texts: list[str]) -> list[list[float]]:
            nonlocal calls
            calls += 1
            return [[0.0]] * len(texts)

        async def embedding_dimension(self) -> int:
            return 1

    from rfnry_rag.ingestion.embeddings.batching import embed_batched

    result = await embed_batched(Fake(), ["x"] * 50, batch_size=100)
    assert len(result) == 50
    assert calls == 1


async def test_embed_batched_invalid_batch_size() -> None:
    """batch_size < 1 must raise ValueError."""
    import pytest

    class Fake:
        model = "fake"

        async def embed(self, texts: list[str]) -> list[list[float]]:
            return []

        async def embedding_dimension(self) -> int:
            return 1

    from rfnry_rag.ingestion.embeddings.batching import embed_batched

    with pytest.raises(ValueError, match="batch_size"):
        await embed_batched(Fake(), ["x"], batch_size=0)
