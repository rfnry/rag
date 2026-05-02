from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

ObservabilityLevel = Literal["debug", "info", "warn", "error"]


class ObservabilityRecord(BaseModel):
    schema_version: int = 1
    at: datetime
    level: ObservabilityLevel
    kind: str
    knowledge_id: str | None = None
    source_id: str | None = None
    query_id: str | None = None
    ingest_id: str | None = None
    message: str = ""
    context: dict[str, Any] = Field(default_factory=dict)
    error_type: str | None = None
    error_message: str | None = None
    traceback: str | None = None

    model_config = {"frozen": True}
