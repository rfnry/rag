from __future__ import annotations

import asyncio
import contextlib
import sys
from pathlib import Path
from typing import Protocol, runtime_checkable

from rfnry_rag.telemetry.record import IngestTelemetryRow, QueryTelemetryRow

TelemetryRow = QueryTelemetryRow | IngestTelemetryRow


@runtime_checkable
class Sink(Protocol):
    async def write(self, row: TelemetryRow) -> None: ...


class NullSink:
    """Drop every row. The only way to silence telemetry."""

    async def write(self, row: TelemetryRow) -> None:
        return None


class RecordingSink:
    """Capture rows in memory. Test-only convenience."""

    def __init__(self) -> None:
        self.rows: list[TelemetryRow] = []

    async def write(self, row: TelemetryRow) -> None:
        self.rows.append(row)


class JsonlStderrSink:
    """Default sink. One JSON line per row on stderr."""

    async def write(self, row: TelemetryRow) -> None:
        line = row.model_dump_json()
        sys.stderr.write(line + "\n")


class JsonlFileSink:
    """Append one JSON line per row to a file."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._lock = asyncio.Lock()

    async def write(self, row: TelemetryRow) -> None:
        line = row.model_dump_json()
        async with self._lock:
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")


class MultiSink:
    """Fan out rows to several sinks. Per-sink errors are logged and skipped."""

    def __init__(self, sinks: list[Sink]) -> None:
        self._sinks = list(sinks)

    async def write(self, row: TelemetryRow) -> None:
        for sink in self._sinks:
            with contextlib.suppress(Exception):
                await sink.write(row)


class SqlAlchemyTelemetrySink:
    """Persist rows via a configured `BaseMetadataStore`.

    Calls `insert_query_telemetry` / `insert_ingest_telemetry` on the store.
    The store's tables are created during its own `initialize()`.
    """

    def __init__(self, metadata_store: object) -> None:
        self._store = metadata_store

    async def write(self, row: TelemetryRow) -> None:
        if isinstance(row, QueryTelemetryRow):
            await self._store.insert_query_telemetry(row)  # type: ignore[attr-defined]
        else:
            await self._store.insert_ingest_telemetry(row)  # type: ignore[attr-defined]
