"""Tests for ``embed_batched`` — owns chunking and concurrency over any conforming ``BaseEmbeddings``."""

from __future__ import annotations

import asyncio

from rfnry_knowledge.ingestion.embeddings.batching import embed_batched
from rfnry_knowledge.providers import EmbeddingResult


class _Fake:
    name = "fake"
    model = "fake"

    def __init__(self) -> None:
        self.calls: list[int] = []
        self.concurrent = 0
        self.max_concurrent = 0

    async def embed(self, texts: list[str]) -> EmbeddingResult:
        self.calls.append(len(texts))
        self.concurrent += 1
        self.max_concurrent = max(self.max_concurrent, self.concurrent)
        try:
            await asyncio.sleep(0)
            return EmbeddingResult(vectors=[[float(i)] for i in range(len(texts))])
        finally:
            self.concurrent -= 1

    async def embedding_dimension(self) -> int:
        return 1


async def test_embed_batched_chunks_large_input() -> None:
    fake = _Fake()
    result = await embed_batched(fake, ["x"] * 500, batch_size=100)
    assert len(result) == 500
    assert len(fake.calls) == 5
    assert all(c == 100 for c in fake.calls)


async def test_embed_batched_overlaps_sub_batches() -> None:
    fake = _Fake()

    async def slow_embed(texts: list[str]) -> EmbeddingResult:
        fake.concurrent += 1
        fake.max_concurrent = max(fake.max_concurrent, fake.concurrent)
        await asyncio.sleep(0.02)
        fake.concurrent -= 1
        return EmbeddingResult(vectors=[[0.0]] * len(texts))

    fake.embed = slow_embed  # type: ignore[method-assign]
    result = await embed_batched(fake, ["x"] * 500, batch_size=100)
    assert len(result) == 500
    assert fake.max_concurrent >= 2, f"expected concurrent calls >= 2, got {fake.max_concurrent}"


async def test_embed_batched_empty_input() -> None:
    fake = _Fake()
    result = await embed_batched(fake, [], batch_size=100)
    assert result == []
    assert fake.calls == []


async def test_embed_batched_within_batch_size_single_call() -> None:
    fake = _Fake()
    result = await embed_batched(fake, ["x"] * 50, batch_size=100)
    assert len(result) == 50
    assert fake.calls == [50]


async def test_embed_batched_invalid_batch_size() -> None:
    import pytest

    fake = _Fake()
    with pytest.raises(ValueError, match="batch_size"):
        await embed_batched(fake, ["x"], batch_size=0)
