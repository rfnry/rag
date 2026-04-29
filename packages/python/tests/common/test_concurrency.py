import pytest

from rfnry_rag.common.concurrency import run_concurrent


async def test_run_concurrent_rejects_out_of_range_concurrency() -> None:
    from unittest.mock import AsyncMock

    with pytest.raises(ValueError, match="concurrency must be >= 1"):
        await run_concurrent([1], fn=AsyncMock(), concurrency=0)
    with pytest.raises(ValueError, match="concurrency must be <= "):
        await run_concurrent([1], fn=AsyncMock(), concurrency=10_000)


async def test_run_concurrent_runs_items() -> None:
    results = await run_concurrent([1, 2, 3], fn=lambda x: _async_double(x), concurrency=2)
    assert results == [2, 4, 6]


async def _async_double(x: int) -> int:
    return x * 2
