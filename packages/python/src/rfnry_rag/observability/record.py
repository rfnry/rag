from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class ObservabilityRecord(BaseModel):
    """Structured event record emitted by `Observability`.

    Core fields are shared verbatim with the rfnry agent SDK to keep the two
    tools' on-the-wire shape identical for downstream consumers (admin UI,
    log shipper, OTel bridge). Only the identity hooks differ.
    """

    schema_version: int = 1
    at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    level: Literal["debug", "info", "warn", "error"]
    kind: str

    knowledge_id: str | None = None
    source_id: str | None = None
    query_id: str | None = None
    ingest_id: str | None = None

    message: str
    context: dict[str, Any] = Field(default_factory=dict)

    error_type: str | None = None
    error_message: str | None = None
    traceback: str | None = None
