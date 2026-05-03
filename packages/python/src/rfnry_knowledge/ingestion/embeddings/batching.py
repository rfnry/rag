from __future__ import annotations

import asyncio

from rfnry_knowledge.providers import BaseEmbeddings, EmbeddingResult, TokenUsage
from rfnry_knowledge.providers.usage import merge_usage, usage_to_int_dict
from rfnry_knowledge.telemetry.context import add_llm_usage

EMBED_BATCH_SIZE = 100

_EMBED_CONCURRENCY = 3


def _normalize(result: EmbeddingResult | list[list[float]]) -> EmbeddingResult:
    if isinstance(result, EmbeddingResult):
        return result
    return EmbeddingResult(vectors=list(result), usage=None)


async def embed_batched(
    embeddings: BaseEmbeddings,
    texts: list[str],
    batch_size: int = EMBED_BATCH_SIZE,
) -> list[list[float]]:
    """Embed ``texts`` in batches of at most ``batch_size``; return concatenated vectors.

    Sub-batches are gathered concurrently (bounded by ``_EMBED_CONCURRENCY``).
    Token usage from each sub-batch is accumulated and recorded against the
    active telemetry row via ``add_llm_usage``; the caller does not need to
    handle ``EmbeddingResult.usage`` directly.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if not texts:
        return []
    if len(texts) <= batch_size:
        result = _normalize(await embeddings.embed(texts))
        _record(embeddings, result.usage)
        return result.vectors

    sub_batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
    sem = asyncio.Semaphore(_EMBED_CONCURRENCY)

    async def embed_one(chunk: list[str]) -> EmbeddingResult:
        async with sem:
            return _normalize(await embeddings.embed(chunk))

    results = await asyncio.gather(*(embed_one(c) for c in sub_batches))
    vectors: list[list[float]] = []
    usages: list[TokenUsage | None] = []
    for r in results:
        vectors.extend(r.vectors)
        usages.append(r.usage)
    if any(u is not None for u in usages):
        _record(embeddings, merge_usage(*usages))
    return vectors


def _record(embeddings: BaseEmbeddings, usage: TokenUsage | None) -> None:
    if not usage:
        return
    add_llm_usage(embeddings.name, embeddings.model, usage_to_int_dict(usage))
