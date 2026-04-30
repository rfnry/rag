"""Shared embedding utilities used by both retrieval and reasoning SDKs."""

from __future__ import annotations

import asyncio

from rfnry_rag.providers import BaseEmbeddings

EMBED_BATCH_SIZE = 100

# Concurrency cap for sub-batch gather in embed_batched. Providers stay well
# below their rate limits; this is the single place that owns concurrency.
_EMBED_CONCURRENCY = 3


async def embed_batched(
    embeddings: BaseEmbeddings,
    texts: list[str],
    batch_size: int = EMBED_BATCH_SIZE,
) -> list[list[float]]:
    """Embed *texts* in batches of at most *batch_size*.

    Most provider APIs accept a bounded number of inputs per call; this helper
    slices the input list and concatenates the vectors in order. Empty input
    returns an empty list (no provider call is made).

    Sub-batches are gathered concurrently (bounded by *_EMBED_CONCURRENCY*).
    Provider-level ``embed()`` implementations should be simple single-call
    methods — concurrency is owned here, not inside individual providers."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if not texts:
        return []
    if len(texts) <= batch_size:
        return await embeddings.embed(texts)

    sub_batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
    sem = asyncio.Semaphore(_EMBED_CONCURRENCY)

    async def embed_one(chunk: list[str]) -> list[list[float]]:
        async with sem:
            return await embeddings.embed(chunk)

    results = await asyncio.gather(*(embed_one(c) for c in sub_batches))
    out: list[list[float]] = []
    for r in results:
        out.extend(r)
    return out
