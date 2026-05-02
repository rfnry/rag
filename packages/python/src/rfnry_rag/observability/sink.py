from __future__ import annotations

import asyncio
import contextlib
import os
import sys
from pathlib import Path
from typing import Any, Protocol, TextIO, runtime_checkable

from pydantic import BaseModel, PrivateAttr

from rfnry_rag.observability.record import ObservabilityLevel, ObservabilityRecord


@runtime_checkable
class ObservabilitySink(Protocol):
    async def emit(self, record: ObservabilityRecord) -> None: ...


class JsonlStderrSink(BaseModel):
    stream: Any = None

    model_config = {"arbitrary_types_allowed": True, "frozen": True}

    async def emit(self, record: ObservabilityRecord) -> None:
        target: TextIO = self.stream if self.stream is not None else sys.stderr
        line = record.model_dump_json() + "\n"
        await asyncio.to_thread(_write_line, target, line)


class JsonlFileSink(BaseModel):
    path: Path

    model_config = {"arbitrary_types_allowed": True}

    _lock: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)

    async def emit(self, record: ObservabilityRecord) -> None:
        line = record.model_dump_json() + "\n"
        async with self._lock:
            await asyncio.to_thread(_append_to_file, self.path, line)


class MultiSink(BaseModel):
    sinks: list[ObservabilitySink]

    model_config = {"arbitrary_types_allowed": True}

    async def emit(self, record: ObservabilityRecord) -> None:
        for sink in self.sinks:
            with contextlib.suppress(Exception):
                await sink.emit(record)


class NullSink(BaseModel):
    model_config = {"frozen": True}

    async def emit(self, record: ObservabilityRecord) -> None:
        return None


_RESET = "\x1b[0m"
_LEVEL_COLOR: dict[ObservabilityLevel, str] = {
    "debug": "\x1b[2;37m",
    "info": "\x1b[36m",
    "warn": "\x1b[33m",
    "error": "\x1b[31m",
}
_LEVEL_TAG: dict[ObservabilityLevel, str] = {
    "debug": "DEBUG",
    "info": "INFO ",
    "warn": "WARN ",
    "error": "ERROR",
}


class PrettyStderrSink(BaseModel):
    stream: Any = None
    use_color: bool = True

    model_config = {"arbitrary_types_allowed": True, "frozen": True}

    async def emit(self, record: ObservabilityRecord) -> None:
        target: TextIO = self.stream if self.stream is not None else sys.stderr
        line = _format_pretty(record, color=self.use_color)
        await asyncio.to_thread(_write_line, target, line)


def default_observability_sink() -> ObservabilitySink:
    if os.environ.get("RFNRY_RAG_OBSERVABILITY_FORMAT") == "json":
        return JsonlStderrSink()
    if os.environ.get("RFNRY_RAG_OBSERVABILITY_FORMAT") == "pretty":
        return PrettyStderrSink(use_color=os.environ.get("NO_COLOR") is None)
    if sys.stderr.isatty():
        return PrettyStderrSink(use_color=os.environ.get("NO_COLOR") is None)
    return JsonlStderrSink()


def _write_line(target: TextIO, line: str) -> None:
    target.write(line)
    target.flush()


def _append_to_file(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line)


def _format_pretty(record: ObservabilityRecord, *, color: bool) -> str:
    tag = _LEVEL_TAG[record.level]
    if color:
        tag = f"{_LEVEL_COLOR[record.level]}{tag}{_RESET}"
    parts = [tag, f"{record.kind:<28}"]
    if record.knowledge_id:
        parts.append(f"knowledge={record.knowledge_id}")
    if record.query_id:
        parts.append(f"query={record.query_id}")
    if record.ingest_id:
        parts.append(f"ingest={record.ingest_id}")
    if record.source_id:
        parts.append(f"source={record.source_id}")
    for k, v in record.context.items():
        if v is None:
            continue
        parts.append(f"{k}={v}")
    if record.message and record.message != record.kind:
        parts.append(record.message)
    line = " ".join(parts)
    if record.error_message:
        line += f"  ({record.error_type}: {record.error_message})"
    line += "\n"
    if record.traceback:
        for tb_line in record.traceback.rstrip().splitlines():
            line += f"  {tb_line}\n"
    return line
