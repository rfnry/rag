from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field


class QueryTelemetryRow(BaseModel):
    schema_version: int = 1
    at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    knowledge_id: str | None = None
    query_id: str

    mode: Literal["indexed", "full_context"]
    routing_decision: str
    corpus_tokens: int = 0

    provider: str | None = None
    model: str | None = None
    stop_reason: str | None = None
    tokens_input: int = 0
    tokens_output: int = 0
    tokens_cache_creation: int = 0
    tokens_cache_read: int = 0
    llm_calls: int = 0

    duration_ms: int = 0
    retrieval_ms: int = 0
    grounding_ms: int = 0
    generation_ms: int = 0

    chunks_retrieved: int = 0
    methods_used: list[str] = Field(default_factory=list)
    method_durations_ms: dict[str, int] = Field(default_factory=dict)
    method_errors: int = 0

    grounding_decision: Literal["grounded", "ungrounded", "clarification"] | None = None
    confidence: float | None = None

    outcome: Literal["success", "refused", "error"]
    error_type: str | None = None
    error_message: str | None = None


class IngestTelemetryRow(BaseModel):
    schema_version: int = 1
    at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    knowledge_id: str | None = None
    source_id: str
    ingest_id: str
    source_type: str | None = None

    chunks_count: int = 0
    pages_count: int = 0

    contextual_chunk_calls: int = 0
    contextual_chunk_skipped: bool = False
    document_expansion_calls: int = 0
    document_expansion_chunk_failures: int = 0
    vision_pages_analyzed: int = 0
    vision_pages_skipped: int = 0
    graph_extraction_failed: bool = False

    tokens_input: int = 0
    tokens_output: int = 0
    tokens_cache_creation: int = 0
    tokens_cache_read: int = 0
    llm_calls: int = 0

    duration_ms: int = 0
    parse_ms: int = 0
    chunk_ms: int = 0
    enrichment_ms: int = 0
    embed_ms: int = 0
    persist_ms: int = 0

    outcome: Literal["success", "partial", "error"]
    notes_count: int = 0
    error_type: str | None = None
    error_message: str | None = None
