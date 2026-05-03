from __future__ import annotations

from contextvars import ContextVar, Token
from typing import Any

from rfnry_knowledge.telemetry.record import IngestTelemetryRow, QueryTelemetryRow

TelemetryRow = QueryTelemetryRow | IngestTelemetryRow

_row_var: ContextVar[TelemetryRow | None] = ContextVar("rfnry_knowledge_telemetry_row", default=None)


def current_query_row() -> QueryTelemetryRow | None:
    row = _row_var.get()
    if isinstance(row, QueryTelemetryRow):
        return row
    return None


def current_ingest_row() -> IngestTelemetryRow | None:
    row = _row_var.get()
    if isinstance(row, IngestTelemetryRow):
        return row
    return None


def current_row() -> TelemetryRow | None:
    return _row_var.get()


def _set_row(row: TelemetryRow) -> Token[TelemetryRow | None]:
    return _row_var.set(row)


def _reset_row(token: Token[TelemetryRow | None]) -> None:
    _row_var.reset(token)


def add_llm_usage(provider: str, model: str, usage: dict[str, int]) -> None:
    row = _row_var.get()
    if row is None:
        return
    row.llm_calls += 1
    if isinstance(row, QueryTelemetryRow):
        if row.provider is None:
            row.provider = provider
        if row.model is None:
            row.model = model
    row.tokens_input += int(usage.get("tokens_input", 0))
    row.tokens_output += int(usage.get("tokens_output", 0))
    row.tokens_cache_creation += int(usage.get("tokens_cache_creation", 0))
    row.tokens_cache_read += int(usage.get("tokens_cache_read", 0))


def increment_ingest_field(field: str, n: int = 1) -> None:
    row = current_ingest_row()
    if row is None:
        return
    setattr(row, field, getattr(row, field) + n)


def set_ingest_field(field: str, value: Any) -> None:
    row = current_ingest_row()
    if row is None:
        return
    setattr(row, field, value)
