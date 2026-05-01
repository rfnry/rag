from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from rfnry_rag.observability.record import ObservabilityRecord
from rfnry_rag.observability.sinks import JsonlStderrSink, Sink

_LEVEL_ORDER: dict[str, int] = {"debug": 10, "info": 20, "warn": 30, "error": 40}


@dataclass
class Observability:
    """Structured-event emitter. Always-on; pass `NullSink()` to silence."""

    sink: Sink = field(default_factory=JsonlStderrSink)
    level: Literal["debug", "info", "warn", "error"] = "info"

    async def emit(
        self,
        level: Literal["debug", "info", "warn", "error"],
        kind: str,
        message: str = "",
        *,
        knowledge_id: str | None = None,
        source_id: str | None = None,
        query_id: str | None = None,
        ingest_id: str | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
        traceback: str | None = None,
        **context: Any,
    ) -> None:
        if _LEVEL_ORDER[level] < _LEVEL_ORDER[self.level]:
            return
        record = ObservabilityRecord(
            level=level,
            kind=kind,
            message=message,
            knowledge_id=knowledge_id,
            source_id=source_id,
            query_id=query_id,
            ingest_id=ingest_id,
            error_type=error_type,
            error_message=error_message,
            traceback=traceback,
            context=context,
        )
        await self.sink.emit(record)
