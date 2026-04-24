"""Tests that each embedding provider chunks large inputs within its API batch limit."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from rfnry_rag.common.language_model import LanguageModelProvider
from rfnry_rag.retrieval.modules.ingestion.embeddings.cohere import (
    _COHERE_MAX_BATCH,
    _CohereEmbeddings,
)
from rfnry_rag.retrieval.modules.ingestion.embeddings.openai import (
    _OPENAI_MAX_BATCH,
    _OpenAIEmbeddings,
)
from rfnry_rag.retrieval.modules.ingestion.embeddings.voyage import (
    _VOYAGE_MAX_BATCH,
    _VoyageEmbeddings,
)


async def test_openai_embeddings_chunks_large_batch() -> None:
    """3000 inputs → two API calls: 2048 then 952."""
    calls: list[int] = []

    async def fake_create(input: list[str], **kwargs: object) -> object:
        n = len(input)
        calls.append(n)
        return SimpleNamespace(data=[SimpleNamespace(embedding=[0.0]) for _ in range(n)])

    with patch("rfnry_rag.retrieval.modules.ingestion.embeddings.openai.AsyncOpenAI") as mock_cls:
        fake_client = MagicMock()
        fake_client.embeddings.create = AsyncMock(side_effect=fake_create)
        mock_cls.return_value = fake_client

        provider = LanguageModelProvider(provider="openai", model="text-embedding-3-small", api_key="sk-test")
        emb = _OpenAIEmbeddings(provider=provider)
        result = await emb.embed(["x"] * 3000)

    assert calls == [2048, 952], f"expected [2048, 952], got {calls}"
    assert len(result) == 3000


async def test_openai_embeddings_within_limit_single_call() -> None:
    """Inputs at or below the cap must be sent in a single API call."""
    calls: list[int] = []

    async def fake_create(input: list[str], **kwargs: object) -> object:
        n = len(input)
        calls.append(n)
        return SimpleNamespace(data=[SimpleNamespace(embedding=[0.0]) for _ in range(n)])

    with patch("rfnry_rag.retrieval.modules.ingestion.embeddings.openai.AsyncOpenAI") as mock_cls:
        fake_client = MagicMock()
        fake_client.embeddings.create = AsyncMock(side_effect=fake_create)
        mock_cls.return_value = fake_client

        provider = LanguageModelProvider(provider="openai", model="text-embedding-3-small", api_key="sk-test")
        emb = _OpenAIEmbeddings(provider=provider)
        result = await emb.embed(["x"] * _OPENAI_MAX_BATCH)

    assert calls == [2048], f"expected single call of 2048, got {calls}"
    assert len(result) == _OPENAI_MAX_BATCH


async def test_voyage_embeddings_chunks_large_batch() -> None:
    """300 inputs → three API calls: 128, 128, 44."""
    calls: list[int] = []

    async def fake_embed(texts: list[str], **kwargs: object) -> object:
        calls.append(len(texts))
        return SimpleNamespace(embeddings=[[0.0] for _ in texts])

    with patch("rfnry_rag.retrieval.modules.ingestion.embeddings.voyage.voyageai") as mock_voyageai:
        fake_client = MagicMock()
        fake_client.embed = AsyncMock(side_effect=fake_embed)
        mock_voyageai.AsyncClient.return_value = fake_client

        provider = LanguageModelProvider(provider="voyage", model="voyage-3", api_key="vo-test")
        emb = _VoyageEmbeddings(provider=provider)
        result = await emb.embed(["x"] * 300)

    assert calls == [128, 128, 44], f"expected [128, 128, 44], got {calls}"
    assert len(result) == 300


async def test_voyage_embeddings_within_limit_single_call() -> None:
    """Inputs at or below the cap must be sent in a single API call."""
    calls: list[int] = []

    async def fake_embed(texts: list[str], **kwargs: object) -> object:
        calls.append(len(texts))
        return SimpleNamespace(embeddings=[[0.0] for _ in texts])

    with patch("rfnry_rag.retrieval.modules.ingestion.embeddings.voyage.voyageai") as mock_voyageai:
        fake_client = MagicMock()
        fake_client.embed = AsyncMock(side_effect=fake_embed)
        mock_voyageai.AsyncClient.return_value = fake_client

        provider = LanguageModelProvider(provider="voyage", model="voyage-3", api_key="vo-test")
        emb = _VoyageEmbeddings(provider=provider)
        result = await emb.embed(["x"] * _VOYAGE_MAX_BATCH)

    assert calls == [128], f"expected single call of 128, got {calls}"
    assert len(result) == _VOYAGE_MAX_BATCH


async def test_cohere_embeddings_chunks_large_batch() -> None:
    """200 inputs → three API calls: 96, 96, 8."""
    calls: list[int] = []

    async def fake_embed(texts: list[str], **kwargs: object) -> object:
        n = len(texts)
        calls.append(n)
        embeddings_ns = SimpleNamespace(float_=[[0.0] for _ in range(n)])
        return SimpleNamespace(embeddings=embeddings_ns)

    with patch("rfnry_rag.retrieval.modules.ingestion.embeddings.cohere.cohere") as mock_cohere:
        fake_client = MagicMock()
        fake_client.embed = AsyncMock(side_effect=fake_embed)
        mock_cohere.AsyncClientV2.return_value = fake_client

        provider = LanguageModelProvider(provider="cohere", model="embed-english-v3.0", api_key="co-test")
        emb = _CohereEmbeddings(provider=provider)
        result = await emb.embed(["x"] * 200)

    assert calls == [96, 96, 8], f"expected [96, 96, 8], got {calls}"
    assert len(result) == 200


async def test_cohere_embeddings_within_limit_single_call() -> None:
    """Inputs at or below the cap must be sent in a single API call."""
    calls: list[int] = []

    async def fake_embed(texts: list[str], **kwargs: object) -> object:
        n = len(texts)
        calls.append(n)
        embeddings_ns = SimpleNamespace(float_=[[0.0] for _ in range(n)])
        return SimpleNamespace(embeddings=embeddings_ns)

    with patch("rfnry_rag.retrieval.modules.ingestion.embeddings.cohere.cohere") as mock_cohere:
        fake_client = MagicMock()
        fake_client.embed = AsyncMock(side_effect=fake_embed)
        mock_cohere.AsyncClientV2.return_value = fake_client

        provider = LanguageModelProvider(provider="cohere", model="embed-english-v3.0", api_key="co-test")
        emb = _CohereEmbeddings(provider=provider)
        result = await emb.embed(["x"] * _COHERE_MAX_BATCH)

    assert calls == [96], f"expected single call of 96, got {calls}"
    assert len(result) == _COHERE_MAX_BATCH
