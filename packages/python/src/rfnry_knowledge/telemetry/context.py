from __future__ import annotations

from contextvars import ContextVar, Token
from typing import Any

from pydantic import BaseModel

from rfnry_knowledge.telemetry.record import (
    IngestTelemetryRow,
    MemoryAddTelemetryRow,
    MemoryUpdateTelemetryRow,
    QueryTelemetryRow,
)

TelemetryRow = QueryTelemetryRow | IngestTelemetryRow

# Rows that carry LLM token-usage fields (llm_calls, tokens_*).
LLMUsageRow = QueryTelemetryRow | IngestTelemetryRow | MemoryAddTelemetryRow | MemoryUpdateTelemetryRow

_row_var: ContextVar[BaseModel | None] = ContextVar("rfnry_knowledge_telemetry_row", default=None)


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
    row = _row_var.get()
    if isinstance(row, (QueryTelemetryRow, IngestTelemetryRow)):
        return row
    return None


def _set_row(row: BaseModel) -> Token[BaseModel | None]:
    return _row_var.set(row)


def _reset_row(token: Token[BaseModel | None]) -> None:
    _row_var.reset(token)


def add_llm_usage(provider: str, model: str, usage: dict[str, int]) -> None:
    row = _row_var.get()
    if not isinstance(row, (QueryTelemetryRow, IngestTelemetryRow, MemoryAddTelemetryRow, MemoryUpdateTelemetryRow)):
        return
    row.llm_calls += 1
    if isinstance(row, QueryTelemetryRow):
        if row.provider is None:
            row.provider = provider
        if row.model is None:
            row.model = model
    row.tokens_input += int(usage.get("input", usage.get("tokens_input", 0)) or 0)
    row.tokens_output += int(usage.get("output", usage.get("tokens_output", 0)) or 0)
    row.tokens_cache_creation += int(usage.get("cache_creation", usage.get("tokens_cache_creation", 0)) or 0)
    row.tokens_cache_read += int(usage.get("cache_read", usage.get("tokens_cache_read", 0)) or 0)


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
