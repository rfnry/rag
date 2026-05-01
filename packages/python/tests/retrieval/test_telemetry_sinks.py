from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from rfnry_rag.stores.metadata.sqlalchemy import SQLAlchemyMetadataStore
from rfnry_rag.telemetry import (
    IngestTelemetryRow,
    JsonlFileSink,
    JsonlStderrSink,
    MultiSink,
    NullSink,
    QueryTelemetryRow,
    RecordingSink,
    SqlAlchemyTelemetrySink,
    Telemetry,
)
from rfnry_rag.telemetry.context import (
    _reset_row,
    _set_row,
    add_llm_usage,
    current_ingest_row,
    current_query_row,
)


def _make_query_row(**overrides) -> QueryTelemetryRow:
    base = dict(
        query_id="q-1",
        mode="indexed",
        routing_decision="indexed",
        outcome="success",
    )
    base.update(overrides)
    return QueryTelemetryRow(**base)


def _make_ingest_row(**overrides) -> IngestTelemetryRow:
    base = dict(
        source_id="s-1",
        ingest_id="i-1",
        outcome="success",
    )
    base.update(overrides)
    return IngestTelemetryRow(**base)


@pytest.mark.asyncio
async def test_recording_sink_captures_both_row_types() -> None:
    sink = RecordingSink()
    await sink.write(_make_query_row())
    await sink.write(_make_ingest_row())
    assert len(sink.rows) == 2
    assert isinstance(sink.rows[0], QueryTelemetryRow)
    assert isinstance(sink.rows[1], IngestTelemetryRow)


@pytest.mark.asyncio
async def test_null_sink_swallows() -> None:
    sink = NullSink()
    await sink.write(_make_query_row())


@pytest.mark.asyncio
async def test_multisink_fans_out_and_isolates_failures() -> None:
    class _Boom:
        async def write(self, row):
            raise RuntimeError("boom")

    captured = RecordingSink()
    multi = MultiSink([_Boom(), captured])
    await multi.write(_make_query_row())
    assert len(captured.rows) == 1


@pytest.mark.asyncio
async def test_jsonl_stderr_emits_query_row(capsys) -> None:
    sink = JsonlStderrSink()
    await sink.write(_make_query_row(knowledge_id="k", tokens_input=12))
    line = capsys.readouterr().err.strip().splitlines()[-1]
    parsed = json.loads(line)
    assert parsed["mode"] == "indexed"
    assert parsed["tokens_input"] == 12


@pytest.mark.asyncio
async def test_jsonl_file_sink_appends(tmp_path) -> None:
    path = tmp_path / "tel.jsonl"
    sink = JsonlFileSink(path)
    await sink.write(_make_query_row(query_id="a"))
    await sink.write(_make_ingest_row(source_id="b"))
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["query_id"] == "a"
    assert json.loads(lines[1])["source_id"] == "b"


@pytest.mark.asyncio
async def test_telemetry_default_sink_is_jsonl_stderr() -> None:
    tel = Telemetry()
    assert isinstance(tel.sink, JsonlStderrSink)


@pytest.mark.asyncio
async def test_sqlalchemy_query_telemetry_roundtrip(tmp_path) -> None:
    store = SQLAlchemyMetadataStore(f"sqlite:///{tmp_path}/m.db")
    await store.initialize()
    try:
        row = _make_query_row(
            knowledge_id="k1",
            tokens_input=11,
            tokens_output=22,
            methods_used=["vector", "graph"],
            method_durations_ms={"vector": 30, "graph": 40},
        )
        await store.insert_query_telemetry(row)
        rows = await store.list_query_telemetry(knowledge_id="k1")
        assert len(rows) == 1
        got = rows[0]
        assert got.query_id == "q-1"
        assert got.tokens_input == 11
        assert got.tokens_output == 22
        assert got.methods_used == ["vector", "graph"]
        assert got.method_durations_ms == {"vector": 30, "graph": 40}
    finally:
        await store.shutdown()


@pytest.mark.asyncio
async def test_sqlalchemy_ingest_telemetry_roundtrip(tmp_path) -> None:
    store = SQLAlchemyMetadataStore(f"sqlite:///{tmp_path}/m.db")
    await store.initialize()
    try:
        row = _make_ingest_row(
            knowledge_id="k1",
            chunks_count=42,
            outcome="partial",
            notes_count=2,
            graph_extraction_failed=True,
        )
        await store.insert_ingest_telemetry(row)
        rows = await store.list_ingest_telemetry(knowledge_id="k1")
        assert len(rows) == 1
        got = rows[0]
        assert got.outcome == "partial"
        assert got.chunks_count == 42
        assert got.graph_extraction_failed is True
        assert got.notes_count == 2
    finally:
        await store.shutdown()


@pytest.mark.asyncio
async def test_sqlalchemy_list_filters_by_date(tmp_path) -> None:
    store = SQLAlchemyMetadataStore(f"sqlite:///{tmp_path}/m.db")
    await store.initialize()
    try:
        old = datetime.now(UTC) - timedelta(days=2)
        recent = datetime.now(UTC)
        await store.insert_query_telemetry(_make_query_row(query_id="old", at=old))
        await store.insert_query_telemetry(_make_query_row(query_id="new", at=recent))
        cutoff = datetime.now(UTC) - timedelta(days=1)
        rows = await store.list_query_telemetry(since=cutoff)
        assert [r.query_id for r in rows] == ["new"]
    finally:
        await store.shutdown()


@pytest.mark.asyncio
async def test_sqlalchemy_list_respects_limit(tmp_path) -> None:
    store = SQLAlchemyMetadataStore(f"sqlite:///{tmp_path}/m.db")
    await store.initialize()
    try:
        for i in range(5):
            await store.insert_query_telemetry(_make_query_row(query_id=f"q{i}"))
        rows = await store.list_query_telemetry(limit=2)
        assert len(rows) == 2
    finally:
        await store.shutdown()


@pytest.mark.asyncio
async def test_sqlalchemy_telemetry_sink_persists(tmp_path) -> None:
    store = SQLAlchemyMetadataStore(f"sqlite:///{tmp_path}/m.db")
    await store.initialize()
    try:
        sink = SqlAlchemyTelemetrySink(store)
        await sink.write(_make_query_row())
        await sink.write(_make_ingest_row())
        q_rows = await store.list_query_telemetry()
        i_rows = await store.list_ingest_telemetry()
        assert len(q_rows) == 1
        assert len(i_rows) == 1
    finally:
        await store.shutdown()


@pytest.mark.asyncio
async def test_add_llm_usage_accumulates_on_query_row() -> None:
    row = _make_query_row()
    token = _set_row(row)
    try:
        add_llm_usage(
            "anthropic",
            "claude-3-5-sonnet",
            {
                "tokens_input": 10,
                "tokens_output": 20,
                "tokens_cache_creation": 1,
                "tokens_cache_read": 2,
            },
        )
        add_llm_usage("anthropic", "claude-3-5-sonnet", {"tokens_input": 5})
        assert current_query_row() is row
        assert row.llm_calls == 2
        assert row.tokens_input == 15
        assert row.tokens_output == 20
        assert row.tokens_cache_creation == 1
        assert row.tokens_cache_read == 2
        assert row.provider == "anthropic"
        assert row.model == "claude-3-5-sonnet"
    finally:
        _reset_row(token)


@pytest.mark.asyncio
async def test_add_llm_usage_accumulates_on_ingest_row() -> None:
    row = _make_ingest_row()
    token = _set_row(row)
    try:
        add_llm_usage("openai", "gpt-4o", {"tokens_input": 7, "tokens_output": 3})
        assert current_ingest_row() is row
        assert row.llm_calls == 1
        assert row.tokens_input == 7
        assert row.tokens_output == 3
    finally:
        _reset_row(token)


@pytest.mark.asyncio
async def test_add_llm_usage_no_op_outside_context() -> None:
    add_llm_usage("anthropic", "claude", {"tokens_input": 99})
