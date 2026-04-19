import asyncio
from collections.abc import Awaitable, Callable, Iterable


async def run_concurrent[T, R](
    items: Iterable[T],
    fn: Callable[[T], Awaitable[R]],
    concurrency: int,
) -> list[R]:
    """Run fn(item) for each item with bounded concurrency. Results preserve input order."""
    semaphore = asyncio.Semaphore(concurrency)

    async def _run(item: T) -> R:
        async with semaphore:
            return await fn(item)

    tasks = [asyncio.create_task(_run(item)) for item in items]
    return list(await asyncio.gather(*tasks))
