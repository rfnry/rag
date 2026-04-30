from __future__ import annotations

import asyncio
import hashlib
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from rfnry_rag.ingestion.embeddings.base import BaseEmbeddings
from rfnry_rag.ingestion.embeddings.batching import embed_batched
from rfnry_rag.logging import get_logger
from rfnry_rag.models import Source, VectorPoint
from rfnry_rag.stores.metadata.base import BaseMetadataStore
from rfnry_rag.stores.vector.base import BaseVectorStore

logger = get_logger("chunk/ingestion/batch")


@dataclass
class TextRecord:
    """A single record for batch ingestion."""

    text: str
    title: str
    knowledge_id: str
    source_type: str = "document"
    metadata: dict | None = None


@dataclass
class BatchError:
    """A single error encountered during batch ingestion."""

    record_index: int
    title: str
    error: str


@dataclass
class BatchProgress:
    """Progress update emitted during batch ingestion."""

    processed: int
    total: int
    succeeded: int
    failed: int
    skipped_duplicates: int


@dataclass
class BatchStats:
    """Final statistics after batch ingestion completes."""

    total: int = 0
    succeeded: int = 0
    failed: int = 0
    skipped_duplicates: int = 0
    duration_seconds: float = 0.0
    errors: list[BatchError] = field(default_factory=list)


@dataclass
class BatchConfig:
    """Configuration for batch ingestion."""

    batch_size: int = 100
    concurrency: int = 5
    skip_duplicates: bool = True
    on_progress: Callable[[BatchProgress], None] | None = None

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.batch_size > 100_000:
            raise ValueError(f"batch_size must be <= 100_000, got {self.batch_size}")
        if self.concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        if self.concurrency > 20:
            raise ValueError("concurrency must be <= 20 — higher values risk overwhelming the vector store")


class BatchIngestionService:
    """Bulk ingestion of text records with batched embedding and vector upserts."""

    def __init__(
        self,
        embeddings: BaseEmbeddings,
        vector_store: BaseVectorStore,
        embedding_model_name: str,
        config: BatchConfig | None = None,
        metadata_store: BaseMetadataStore | None = None,
        on_complete: Callable[[str | None], Awaitable[Any]] | None = None,
    ) -> None:
        self._embeddings = embeddings
        self._vector_store = vector_store
        self._metadata_store = metadata_store
        self._embedding_model_name = embedding_model_name
        self._config = config or BatchConfig()
        self._on_complete = on_complete

    async def ingest_batch(self, records: list[TextRecord]) -> BatchStats:
        """Ingest a list of records."""

        async def _iter():
            for r in records:
                yield r

        return await self.ingest_stream(_iter(), total=len(records))

    async def ingest_stream(
        self,
        stream: AsyncIterator[TextRecord],
        total: int | None = None,
    ) -> BatchStats:
        """Ingest records from an async iterator in batches."""
        start = time.monotonic()
        stats = BatchStats()
        seen_keys: set[str] = set()
        semaphore = asyncio.Semaphore(self._config.concurrency)
        batch: list[tuple[int, TextRecord]] = []
        knowledge_ids_touched: set[str] = set()

        async def _process_batch(items: list[tuple[int, TextRecord]]) -> None:
            async with semaphore:
                await self._process_batch(items, stats, seen_keys, knowledge_ids_touched)
                if self._config.on_progress:
                    self._config.on_progress(
                        BatchProgress(
                            processed=stats.succeeded + stats.failed + stats.skipped_duplicates,
                            total=total or 0,
                            succeeded=stats.succeeded,
                            failed=stats.failed,
                            skipped_duplicates=stats.skipped_duplicates,
                        )
                    )

        # Bounded inflight queue: the semaphore throttles concurrent *execution*
        # but we also need to throttle *task creation* so a million-record stream
        # does not schedule a million task objects at once.
        max_inflight = max(self._config.concurrency * 2, 2)
        in_progress: set[asyncio.Task] = set()

        async def _schedule(current_batch: list[tuple[int, TextRecord]]) -> None:
            if len(in_progress) >= max_inflight:
                done, _ = await asyncio.wait(in_progress, return_when=asyncio.FIRST_COMPLETED)
                for t in done:
                    in_progress.discard(t)
                    # surface exceptions eagerly; otherwise failures would only
                    # appear at final gather and mask which batch failed
                    exc = t.exception()
                    if exc is not None:
                        raise exc
            task = asyncio.create_task(_process_batch(current_batch))
            in_progress.add(task)

        index = 0
        async for record in stream:
            batch.append((index, record))
            index += 1
            if len(batch) >= self._config.batch_size:
                await _schedule(batch[:])
                batch = []

        if batch:
            await _schedule(batch)

        if in_progress:
            # return_exceptions=True so a single failure doesn't cancel the rest.
            # Without it, gather propagates the first exception, cancels other
            # in-flight tasks, and stats.total would be inconsistent with actual
            # work done. Mirrors the eager-surface logic above.
            results = await asyncio.gather(*in_progress, return_exceptions=True)
            for result in results:
                if isinstance(result, BaseException):
                    logger.exception("batch ingestion task failed", exc_info=result)

        stats.total = index
        stats.duration_seconds = time.monotonic() - start

        if self._on_complete:
            # SERIAL: on_complete invalidates a BM25 cache keyed by knowledge_id;
            # parallel calls for distinct knowledge_ids would be safe, but the
            # callback implementation is not documented as concurrency-safe, and
            # the set is typically tiny (≤ handful of knowledge_ids per batch).
            for kid in knowledge_ids_touched:
                await self._on_complete(kid)

        logger.info(
            "batch ingestion complete: %d total, %d succeeded, %d failed, %d skipped",
            stats.total,
            stats.succeeded,
            stats.failed,
            stats.skipped_duplicates,
        )
        return stats

    async def _process_batch(
        self,
        items: list[tuple[int, TextRecord]],
        stats: BatchStats,
        seen_keys: set[str],
        knowledge_ids_touched: set[str],
    ) -> None:
        """Process a single batch: filter duplicates, embed, upsert."""
        filtered: list[tuple[int, TextRecord]] = []
        for idx, record in items:
            if self._config.skip_duplicates:
                key = f"{record.knowledge_id}:{record.title}"
                if key in seen_keys:
                    stats.skipped_duplicates += 1
                    continue
                seen_keys.add(key)
            filtered.append((idx, record))

        if not filtered:
            return

        texts = [r.text for _, r in filtered]
        try:
            vectors = await embed_batched(self._embeddings, texts)
        except Exception as exc:
            logger.exception("embedding failed for batch of %d records", len(filtered))
            for idx, record in filtered:
                stats.failed += 1
                stats.errors.append(BatchError(record_index=idx, title=record.title, error=str(exc)))
            return

        points: list[VectorPoint] = []
        sources: list[Source] = []
        for i, (_idx, record) in enumerate(filtered):
            source_id = f"batch-{uuid.uuid4().hex[:12]}"
            chunk_id = f"{source_id}-0"
            knowledge_ids_touched.add(record.knowledge_id)
            metadata = record.metadata or {}
            text_hash = hashlib.sha256(record.text.encode()).hexdigest()

            point = VectorPoint(
                point_id=chunk_id,
                vector=vectors[i],
                payload={
                    "source_id": source_id,
                    "content": record.text,
                    "knowledge_id": record.knowledge_id,
                    "source_type": record.source_type,
                    "source_weight": 1.0,
                    "chunk_index": 0,
                    "tags": [],
                    **{f"meta_{k}": v for k, v in metadata.items()},
                },
            )
            points.append(point)

            source = Source(
                source_id=source_id,
                metadata={"title": record.title, **metadata},
                chunk_count=1,
                embedding_model=self._embedding_model_name,
                file_hash=text_hash,
                knowledge_id=record.knowledge_id,
                source_type=record.source_type,
            )
            sources.append(source)

        try:
            if points:
                await self._vector_store.initialize(len(points[0].vector))
                await self._vector_store.upsert(points)
            if self._metadata_store:
                # SERIAL: per-source exception isolation — a failure on one row
                # must not abort the remaining creates. asyncio.gather with
                # return_exceptions=True would also work but adds indirection;
                # the per-iteration try/except is clearer for this error pattern.
                for source in sources:
                    try:
                        await self._metadata_store.create_source(source)
                    except Exception:
                        logger.warning("metadata store failed for source=%s, skipping", source.source_id)
            stats.succeeded += len(filtered)
        except Exception as exc:
            logger.exception("vector upsert failed for batch of %d records", len(filtered))
            for idx, record in filtered:
                stats.failed += 1
                stats.errors.append(BatchError(record_index=idx, title=record.title, error=str(exc)))
