"""Per-method retrieval/ingestion timings: methods record their durations on
the active row and emit retrieval.method.* / ingestion.method.* events.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from rfnry_rag.ingestion.methods.document import DocumentIngestion
from rfnry_rag.ingestion.methods.vector import VectorIngestion
from rfnry_rag.models import VectorResult
from rfnry_rag.observability import Observability, RecordingSink
from rfnry_rag.observability.context import _reset_obs, _set_obs
from rfnry_rag.retrieval.methods.document import DocumentRetrieval
from rfnry_rag.retrieval.methods.vector import VectorRetrieval
from rfnry_rag.telemetry import IngestTelemetryRow, QueryTelemetryRow
from rfnry_rag.telemetry.context import _reset_row, _set_row


def _query_row() -> QueryTelemetryRow:
    return QueryTelemetryRow(query_id="q-1", mode="indexed", routing_decision="indexed", outcome="success")


def _ingest_row() -> IngestTelemetryRow:
    return IngestTelemetryRow(source_id="s-1", ingest_id="i-1", outcome="success")


@pytest.mark.asyncio
async def test_vector_retrieval_records_method_duration_and_event() -> None:
    obs_sink = RecordingSink()
    obs_token = _set_obs(Observability(sink=obs_sink))
    row = _query_row()
    row_token = _set_row(row)

    fake_store = AsyncMock()
    fake_store.search = AsyncMock(
        return_value=[VectorResult(point_id="p1", score=0.9, payload={"content": "x", "source_id": "s"})]
    )
    fake_embeddings = AsyncMock()
    fake_embeddings.embed = AsyncMock(return_value=[[0.0]])

    retriever = VectorRetrieval(store=fake_store, embeddings=fake_embeddings)
    try:
        results = await retriever.search("q", top_k=5)
        assert len(results) == 1
        assert "vector" in row.methods_used
        assert "vector" in row.method_durations_ms
        assert row.method_durations_ms["vector"] >= 0
        assert row.chunks_retrieved == 1
        kinds = [r.kind for r in obs_sink.records]
        assert "retrieval.method.success" in kinds
    finally:
        _reset_row(row_token)
        _reset_obs(obs_token)


@pytest.mark.asyncio
async def test_vector_retrieval_failure_increments_method_errors() -> None:
    obs_sink = RecordingSink()
    obs_token = _set_obs(Observability(sink=obs_sink))
    row = _query_row()
    row_token = _set_row(row)

    fake_store = AsyncMock()
    fake_store.search = AsyncMock(side_effect=RuntimeError("store down"))
    fake_embeddings = AsyncMock()
    fake_embeddings.embed = AsyncMock(return_value=[[0.0]])

    retriever = VectorRetrieval(store=fake_store, embeddings=fake_embeddings)
    try:
        results = await retriever.search("q", top_k=5)
        assert results == []  # error-isolated
        assert row.method_errors == 1
        kinds = [r.kind for r in obs_sink.records]
        assert "retrieval.method.error" in kinds
    finally:
        _reset_row(row_token)
        _reset_obs(obs_token)


@pytest.mark.asyncio
async def test_document_retrieval_records_method_duration_and_event() -> None:
    obs_sink = RecordingSink()
    obs_token = _set_obs(Observability(sink=obs_sink))
    row = _query_row()
    row_token = _set_row(row)

    fake_store = AsyncMock()
    fake_store.search_content = AsyncMock(return_value=[])

    retriever = DocumentRetrieval(store=fake_store)
    try:
        await retriever.search("q")
        assert "document" in row.method_durations_ms
        kinds = [r.kind for r in obs_sink.records]
        assert "retrieval.method.success" in kinds
    finally:
        _reset_row(row_token)
        _reset_obs(obs_token)


@pytest.mark.asyncio
async def test_document_ingestion_emits_method_success() -> None:
    obs_sink = RecordingSink()
    obs_token = _set_obs(Observability(sink=obs_sink))
    row = _ingest_row()
    row_token = _set_row(row)

    fake_store = AsyncMock()
    fake_store.store_content = AsyncMock(return_value=None)

    ing = DocumentIngestion(store=fake_store)
    try:
        await ing.ingest(
            source_id="s-1",
            knowledge_id=None,
            source_type=None,
            source_weight=1.0,
            title="t",
            full_text="body",
            chunks=[],
            tags=[],
            metadata={},
        )
        kinds = [r.kind for r in obs_sink.records]
        assert "ingestion.method.success" in kinds
        assert row.persist_ms >= 0
    finally:
        _reset_row(row_token)
        _reset_obs(obs_token)


@pytest.mark.asyncio
async def test_vector_ingestion_emits_method_event_and_records_duration() -> None:
    obs_sink = RecordingSink()
    obs_token = _set_obs(Observability(sink=obs_sink))
    row = _ingest_row()
    row_token = _set_row(row)

    fake_store = AsyncMock()
    fake_store.upsert = AsyncMock(return_value=None)

    fake_embeddings = SimpleNamespace(
        embed=AsyncMock(return_value=[[0.1, 0.2]]),
        embedding_dimension=AsyncMock(return_value=2),
        name="x",
        model="m",
    )

    from rfnry_rag.ingestion.models import ChunkedContent

    chunk = ChunkedContent(content="body", chunk_index=0)

    ing = VectorIngestion(store=fake_store, embeddings=fake_embeddings, embedding_model_name="x")
    try:
        await ing.ingest(
            source_id="s-1",
            knowledge_id=None,
            source_type=None,
            source_weight=1.0,
            title="t",
            full_text="body",
            chunks=[chunk],
            tags=[],
            metadata={},
        )
        kinds = [r.kind for r in obs_sink.records]
        assert "ingestion.method.success" in kinds
        assert row.embed_ms >= 0
        assert row.chunks_count == 1
    finally:
        _reset_row(row_token)
        _reset_obs(obs_token)
