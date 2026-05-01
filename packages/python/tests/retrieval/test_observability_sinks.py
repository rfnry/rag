from __future__ import annotations

import json

import pytest

from rfnry_rag.observability import (
    JsonlFileSink,
    JsonlStderrSink,
    MultiSink,
    NullSink,
    Observability,
    ObservabilityRecord,
    RecordingSink,
)


def _make_record(**overrides) -> ObservabilityRecord:
    base = dict(level="info", kind="query.start", message="hello")
    base.update(overrides)
    return ObservabilityRecord(**base)


@pytest.mark.asyncio
async def test_recording_sink_captures_records() -> None:
    sink = RecordingSink()
    rec = _make_record()
    await sink.emit(rec)
    await sink.emit(_make_record(level="error", kind="query.error"))
    assert len(sink.records) == 2
    assert sink.records[0].kind == "query.start"
    assert sink.records[1].level == "error"


@pytest.mark.asyncio
async def test_null_sink_swallows() -> None:
    sink = NullSink()
    await sink.emit(_make_record())


@pytest.mark.asyncio
async def test_multisink_fans_out() -> None:
    a = RecordingSink()
    b = RecordingSink()
    multi = MultiSink([a, b])
    rec = _make_record()
    await multi.emit(rec)
    assert len(a.records) == 1
    assert len(b.records) == 1
    assert a.records[0].kind == b.records[0].kind == "query.start"


@pytest.mark.asyncio
async def test_multisink_isolates_failures() -> None:
    class _Boom:
        async def emit(self, record):
            raise RuntimeError("boom")

    captured = RecordingSink()
    multi = MultiSink([_Boom(), captured])
    await multi.emit(_make_record())
    assert len(captured.records) == 1


@pytest.mark.asyncio
async def test_jsonl_stderr_sink_writes_valid_json(capsys) -> None:
    sink = JsonlStderrSink()
    rec = _make_record(context={"k": "v"})
    await sink.emit(rec)
    captured = capsys.readouterr()
    line = captured.err.strip().splitlines()[-1]
    parsed = json.loads(line)
    assert parsed["kind"] == "query.start"
    assert parsed["context"] == {"k": "v"}


@pytest.mark.asyncio
async def test_jsonl_file_sink_appends_lines(tmp_path) -> None:
    path = tmp_path / "obs.jsonl"
    sink = JsonlFileSink(path)
    await sink.emit(_make_record(message="one"))
    await sink.emit(_make_record(message="two"))
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    a = json.loads(lines[0])
    b = json.loads(lines[1])
    assert a["message"] == "one"
    assert b["message"] == "two"


@pytest.mark.asyncio
async def test_observability_filters_by_level() -> None:
    sink = RecordingSink()
    obs = Observability(sink=sink, level="warn")
    await obs.emit("debug", "k", "ignored")
    await obs.emit("info", "k", "ignored")
    await obs.emit("warn", "k", "kept")
    await obs.emit("error", "k", "kept")
    assert [r.message for r in sink.records] == ["kept", "kept"]


@pytest.mark.asyncio
async def test_observability_serialises_context_and_identity() -> None:
    sink = RecordingSink()
    obs = Observability(sink=sink, level="info")
    await obs.emit(
        "info",
        "provider.call",
        "ok",
        knowledge_id="k",
        query_id="q",
        provider="anthropic",
        tokens_input=12,
    )
    rec = sink.records[0]
    assert rec.knowledge_id == "k"
    assert rec.query_id == "q"
    assert rec.context == {"provider": "anthropic", "tokens_input": 12}


@pytest.mark.asyncio
async def test_observability_record_round_trips_json() -> None:
    rec = _make_record(context={"a": 1}, knowledge_id="k", query_id="q")
    payload = rec.model_dump_json()
    parsed = ObservabilityRecord.model_validate_json(payload)
    assert parsed.kind == rec.kind
    assert parsed.context == {"a": 1}
    assert parsed.query_id == "q"


@pytest.mark.asyncio
async def test_observability_default_sink_is_jsonl_stderr() -> None:
    obs = Observability()
    assert isinstance(obs.sink, JsonlStderrSink)
