from __future__ import annotations

import asyncio
import contextlib
import sys
from pathlib import Path
from typing import Protocol, runtime_checkable

from rfnry_rag.observability.record import ObservabilityRecord


@runtime_checkable
class Sink(Protocol):
    async def emit(self, record: ObservabilityRecord) -> None: ...


class NullSink:
    """Drop every record. The only way to silence observability."""

    async def emit(self, record: ObservabilityRecord) -> None:
        return None


class RecordingSink:
    """Capture records in memory. Test-only convenience."""

    def __init__(self) -> None:
        self.records: list[ObservabilityRecord] = []

    async def emit(self, record: ObservabilityRecord) -> None:
        self.records.append(record)


class JsonlStderrSink:
    """Default sink. One JSON line per record on stderr."""

    async def emit(self, record: ObservabilityRecord) -> None:
        line = record.model_dump_json()
        sys.stderr.write(line + "\n")


class JsonlFileSink:
    """Append one JSON line per record to a file."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._lock = asyncio.Lock()

    async def emit(self, record: ObservabilityRecord) -> None:
        line = record.model_dump_json()
        async with self._lock:
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")


class MultiSink:
    """Fan out records to several sinks. Per-sink errors are logged and skipped."""

    def __init__(self, sinks: list[Sink]) -> None:
        self._sinks = list(sinks)

    async def emit(self, record: ObservabilityRecord) -> None:
        for sink in self._sinks:
            with contextlib.suppress(Exception):
                await sink.emit(record)
