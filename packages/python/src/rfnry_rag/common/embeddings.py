"""Shared embedding utilities used by both retrieval and reasoning SDKs."""

from __future__ import annotations

from rfnry_rag.common.protocols import BaseEmbeddings

EMBED_BATCH_SIZE = 100


async def embed_batched(
    embeddings: BaseEmbeddings,
    texts: list[str],
    batch_size: int = EMBED_BATCH_SIZE,
) -> list[list[float]]:
    """Embed *texts* in batches of at most *batch_size*.

    Most provider APIs accept a bounded number of inputs per call; this helper
    slices the input list and concatenates the vectors in order. Empty input
    returns an empty list (no provider call is made)."""
    if not texts:
        return []
    if len(texts) <= batch_size:
        return await embeddings.embed(texts)
    all_vectors: list[list[float]] = []
    # SERIAL: sub-batches are awaited one at a time. A future C3 task will
    # centralise concurrency here with asyncio.gather + a semaphore so that
    # provider-level rate-limit handling lives in exactly one place.
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        vectors = await embeddings.embed(batch)
        all_vectors.extend(vectors)
    return all_vectors
