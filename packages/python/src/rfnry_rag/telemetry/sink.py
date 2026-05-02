from __future__ import annotations

import asyncio
import contextlib
import sys
from pathlib import Path
from typing import Any, Protocol, TextIO, runtime_checkable

from pydantic import BaseModel, PrivateAttr

from rfnry_rag.telemetry.record import IngestTelemetryRow, QueryTelemetryRow

TelemetryRow = QueryTelemetryRow | IngestTelemetryRow


@runtime_checkable
class TelemetrySink(Protocol):
    async def write(self, row: TelemetryRow) -> None: ...


class JsonlStderrTelemetrySink(BaseModel):
    stream: Any = None

    model_config = {"arbitrary_types_allowed": True, "frozen": True}

    async def write(self, row: TelemetryRow) -> None:
        target: TextIO = self.stream if self.stream is not None else sys.stderr
        line = row.model_dump_json() + "\n"
        await asyncio.to_thread(_write_line, target, line)


class JsonlTelemetrySink(BaseModel):
    path: Path

    model_config = {"arbitrary_types_allowed": True}

    _lock: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)

    async def write(self, row: TelemetryRow) -> None:
        line = row.model_dump_json() + "\n"
        async with self._lock:
            await asyncio.to_thread(_append_to_file, self.path, line)


class MultiTelemetrySink(BaseModel):
    sinks: list[TelemetrySink]

    model_config = {"arbitrary_types_allowed": True}

    async def write(self, row: TelemetryRow) -> None:
        for sink in self.sinks:
            with contextlib.suppress(Exception):
                await sink.write(row)


class NullTelemetrySink(BaseModel):
    model_config = {"frozen": True}

    async def write(self, row: TelemetryRow) -> None:
        return None


class SqlAlchemyTelemetrySink(BaseModel):
    metadata_store: Any

    model_config = {"arbitrary_types_allowed": True, "frozen": True}

    async def write(self, row: TelemetryRow) -> None:
        if isinstance(row, QueryTelemetryRow):
            await self.metadata_store.insert_query_telemetry(row)
        else:
            await self.metadata_store.insert_ingest_telemetry(row)


def _write_line(target: TextIO, line: str) -> None:
    target.write(line)
    target.flush()


def _append_to_file(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line)
