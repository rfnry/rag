from __future__ import annotations

from contextvars import ContextVar, Token
from typing import Any

from rfnry_rag.telemetry.record import IngestTelemetryRow, QueryTelemetryRow

TelemetryRow = QueryTelemetryRow | IngestTelemetryRow

_row_var: ContextVar[TelemetryRow | None] = ContextVar("rfnry_rag_telemetry_row", default=None)


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
    """Accumulate LLM token usage onto whichever row is currently active.

    No-op when no row is active (e.g. user calls `LanguageModelClient.generate_text`
    outside any RagEngine entry point). `provider` / `model` populate only on
    `QueryTelemetryRow` — the ingest row carries no such fields because an ingest
    can fan out across several providers (vision, embeddings, BAML) and naming
    a single one would mislead.
    """
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
    """Add ``n`` to ``IngestTelemetryRow.<field>`` on the active row, or no-op."""
    row = current_ingest_row()
    if row is None:
        return
    setattr(row, field, getattr(row, field) + n)


def set_ingest_field(field: str, value: Any) -> None:
    """Set ``IngestTelemetryRow.<field>`` on the active row, or no-op."""
    row = current_ingest_row()
    if row is None:
        return
    setattr(row, field, value)
