from __future__ import annotations

import contextlib
import traceback as tb_module
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from rfnry_rag.observability.record import ObservabilityLevel, ObservabilityRecord
from rfnry_rag.observability.sink import ObservabilitySink, default_observability_sink

_LEVEL_ORDER: dict[ObservabilityLevel, int] = {
    "debug": 10,
    "info": 20,
    "warn": 30,
    "error": 40,
}


class Observability(BaseModel):
    sink: ObservabilitySink = Field(default_factory=default_observability_sink)
    level: ObservabilityLevel = "info"

    model_config = {"arbitrary_types_allowed": True}

    def _enabled(self, level: ObservabilityLevel) -> bool:
        return _LEVEL_ORDER[level] >= _LEVEL_ORDER[self.level]

    async def emit(
        self,
        kind: str,
        message: str = "",
        *,
        level: ObservabilityLevel = "info",
        knowledge_id: str | None = None,
        source_id: str | None = None,
        query_id: str | None = None,
        ingest_id: str | None = None,
        context: dict[str, Any] | None = None,
        error: BaseException | None = None,
    ) -> None:
        if not self._enabled(level):
            return
        record = ObservabilityRecord(
            at=datetime.now(UTC),
            level=level,
            kind=kind,
            knowledge_id=knowledge_id,
            source_id=source_id,
            query_id=query_id,
            ingest_id=ingest_id,
            message=message,
            context=dict(context) if context else {},
            error_type=type(error).__name__ if error is not None else None,
            error_message=str(error) if error is not None else None,
            traceback=_format_traceback(error) if error is not None else None,
        )
        with contextlib.suppress(Exception):
            await self.sink.emit(record)


def _format_traceback(exc: BaseException) -> str:
    return "".join(tb_module.format_exception(type(exc), exc, exc.__traceback__))
