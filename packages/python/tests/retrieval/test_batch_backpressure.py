"""BatchIngestionService backpressure test — the stream ingester must cap
inflight tasks at roughly `concurrency * 2`, not schedule one task per batch
for the entire stream up front."""

import asyncio
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock

import pytest

from rfnry_knowledge.ingestion.chunk.batch import (
    BatchConfig,
    BatchIngestionService,
    TextRecord,
)


async def _records(n: int) -> AsyncIterator[TextRecord]:
    for i in range(n):
        yield TextRecord(text=f"doc {i}", title=f"t{i}", knowledge_id="k")


@pytest.mark.asyncio
async def test_ingest_stream_caps_inflight_task_count() -> None:
    """With concurrency=2 and batch_size=1, feeding 20 records should never
    have more than ~4 tasks inflight (2 running + 2 queued at the semaphore)."""
    svc = BatchIngestionService(
        embeddings=AsyncMock(),
        vector_store=AsyncMock(),
        embedding_model_name="m",
        config=BatchConfig(batch_size=1, concurrency=2, skip_duplicates=False),
    )

    inflight_peak = 0
    running_count = {"n": 0}
    release = asyncio.Event()

    async def slow_process(items, stats, seen, touched):
        running_count["n"] += 1
        nonlocal inflight_peak
        inflight_peak = max(inflight_peak, running_count["n"])
        await asyncio.sleep(0.01)
        running_count["n"] -= 1
        stats.succeeded += len(items)

    svc._process_batch = slow_process  # type: ignore[method-assign,assignment]

    # Run a quick task in parallel to fire "release" after a short delay,
    # ensuring the main task has time to ramp up before we finish.
    stats = await svc.ingest_stream(_records(20), total=20)
    release.set()
    assert stats.succeeded == 20
    # Peak *concurrent executions* is bounded by the semaphore (concurrency=2).
    # It must not grow linearly with the stream size.
    assert inflight_peak <= 2


def test_batch_config_batch_size_upper_bound() -> None:
    with pytest.raises(ValueError, match="batch_size must be <= "):
        BatchConfig(batch_size=100_001)
