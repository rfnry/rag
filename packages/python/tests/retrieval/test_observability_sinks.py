from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from rfnry_knowledge.observability import (
    JsonlFileSink,
    JsonlStderrSink,
    MultiSink,
    NullSink,
    Observability,
    ObservabilityRecord,
    ObservabilitySink,
    PrettyStderrSink,
    default_observability_sink,
)


class _Capture:
    def __init__(self) -> None:
        self.records: list[ObservabilityRecord] = []

    async def emit(self, record: ObservabilityRecord) -> None:
        self.records.append(record)


def _make_record(**overrides: object) -> ObservabilityRecord:
    base: dict[str, object] = {
        "at": datetime.now(UTC),
        "level": "info",
        "kind": "query.start",
        "message": "hello",
    }
    base.update(overrides)
    return ObservabilityRecord(**base)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_capture_sink_collects_records() -> None:
    sink = _Capture()
    await sink.emit(_make_record())
    await sink.emit(_make_record(level="error", kind="query.error"))
    assert len(sink.records) == 2
    assert sink.records[0].kind == "query.start"
    assert sink.records[1].level == "error"


@pytest.mark.asyncio
async def test_null_sink_swallows() -> None:
    sink = NullSink()
    await sink.emit(_make_record())


@pytest.mark.asyncio
async def test_multisink_fans_out_and_isolates_failures() -> None:
    class _Boom:
        async def emit(self, record: ObservabilityRecord) -> None:
            raise RuntimeError("boom")

    captured = _Capture()
    multi = MultiSink(sinks=[_Boom(), captured])
    await multi.emit(_make_record())
    assert len(captured.records) == 1


@pytest.mark.asyncio
async def test_jsonl_stderr_sink_writes_valid_json(capsys: pytest.CaptureFixture[str]) -> None:
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
    sink = JsonlFileSink(path=path)
    await sink.emit(_make_record(message="one"))
    await sink.emit(_make_record(message="two"))
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["message"] == "one"
    assert json.loads(lines[1])["message"] == "two"


@pytest.mark.asyncio
async def test_observability_filters_by_level() -> None:
    sink = _Capture()
    obs = Observability(sink=sink, level="warn")
    await obs.emit("k", "ignored", level="debug")
    await obs.emit("k", "ignored", level="info")
    await obs.emit("k", "kept", level="warn")
    await obs.emit("k", "kept", level="error")
    assert [r.message for r in sink.records] == ["kept", "kept"]


@pytest.mark.asyncio
async def test_observability_serialises_context_and_identity() -> None:
    sink = _Capture()
    obs = Observability(sink=sink, level="info")
    await obs.emit(
        "provider.call",
        "ok",
        knowledge_id="k",
        query_id="q",
        context={"provider": "anthropic", "tokens_input": 12},
    )
    rec = sink.records[0]
    assert rec.knowledge_id == "k"
    assert rec.query_id == "q"
    assert rec.context == {"provider": "anthropic", "tokens_input": 12}


@pytest.mark.asyncio
async def test_observability_emit_extracts_error_metadata() -> None:
    sink = _Capture()
    obs = Observability(sink=sink, level="info")
    try:
        raise ValueError("nope")
    except ValueError as exc:
        await obs.emit("query.error", "boom", level="error", error=exc)
    rec = sink.records[0]
    assert rec.error_type == "ValueError"
    assert rec.error_message == "nope"
    assert rec.traceback is not None
    assert "ValueError: nope" in rec.traceback


@pytest.mark.asyncio
async def test_observability_record_round_trips_json() -> None:
    rec = _make_record(context={"a": 1}, knowledge_id="k", query_id="q")
    payload = rec.model_dump_json()
    parsed = ObservabilityRecord.model_validate_json(payload)
    assert parsed.kind == rec.kind
    assert parsed.context == {"a": 1}
    assert parsed.query_id == "q"


@pytest.mark.asyncio
async def test_observability_default_sink_is_jsonl_stderr_under_pytest() -> None:
    obs = Observability()
    assert isinstance(obs.sink, JsonlStderrSink)


@pytest.mark.asyncio
async def test_default_sink_factory_honors_format_env(monkeypatch) -> None:
    monkeypatch.setenv("KNWL_OBSERVABILITY_FORMAT", "pretty")
    monkeypatch.delenv("NO_COLOR", raising=False)
    sink = default_observability_sink()
    assert isinstance(sink, PrettyStderrSink)

    monkeypatch.setenv("KNWL_OBSERVABILITY_FORMAT", "json")
    assert isinstance(default_observability_sink(), JsonlStderrSink)


@pytest.mark.asyncio
async def test_pretty_stderr_sink_writes_single_line(capsys: pytest.CaptureFixture[str]) -> None:
    sink = PrettyStderrSink(use_color=False)
    rec = _make_record(
        knowledge_id="k",
        query_id="q",
        context={"provider": "anthropic"},
    )
    await sink.emit(rec)
    captured = capsys.readouterr()
    line = captured.err.strip().splitlines()[-1]
    assert "INFO" in line
    assert "query.start" in line
    assert "knowledge=k" in line
    assert "query=q" in line
    assert "provider=anthropic" in line


def test_observability_sink_protocol_runtime_check() -> None:
    assert isinstance(NullSink(), ObservabilitySink)
    assert isinstance(JsonlStderrSink(), ObservabilitySink)
    assert isinstance(_Capture(), ObservabilitySink)
