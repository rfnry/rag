"""Engine-level observability + telemetry: query/ingest entry points emit
lifecycle events and write rows for every transaction.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from _recording import (
    RecordingObservabilitySink as ObsRecordingSink,
)
from _recording import (
    RecordingObservabilitySink as RecordingSink,
)
from _recording import (
    RecordingTelemetrySink as TelRecordingSink,
)

from rfnry_rag.config import RagEngineConfig, RoutingConfig
from rfnry_rag.config.routing import QueryMode
from rfnry_rag.generation.models import Clarification, QueryResult
from rfnry_rag.models import Source
from rfnry_rag.observability import Observability
from rfnry_rag.server import RagEngine
from rfnry_rag.telemetry import (
    IngestTelemetryRow,
    QueryTelemetryRow,
    Telemetry,
)


def _query_result(**overrides) -> QueryResult:
    base = dict(answer="ok", sources=[], grounded=True, confidence=0.9)
    base.update(overrides)
    return QueryResult(**base)


def _build_engine(
    *,
    obs_sink: ObsRecordingSink,
    tel_sink: TelRecordingSink,
    routing: RoutingConfig | None = None,
    query_result: QueryResult | None = None,
    raise_exc: BaseException | None = None,
) -> RagEngine:
    config = MagicMock(spec=RagEngineConfig)
    config.retrieval = SimpleNamespace(history_window=3)
    config.routing = routing or RoutingConfig(mode=QueryMode.INDEXED)
    engine = RagEngine.__new__(RagEngine)
    engine._config = config
    engine._observability = Observability(sink=obs_sink)
    engine._telemetry = Telemetry(sink=tel_sink)
    engine._initialized = True
    engine._retrieval_service = AsyncMock()
    engine._retrieval_service.retrieve = AsyncMock(return_value=([], None))
    engine._structured_retrieval = None
    engine._generation_service = AsyncMock()
    if raise_exc is not None:
        engine._generation_service.generate = AsyncMock(side_effect=raise_exc)
    else:
        engine._generation_service.generate = AsyncMock(return_value=query_result or _query_result())
    engine._knowledge_manager = None
    engine._ingestion_service = None
    engine._structured_ingestion = None
    engine._retrieval_namespace = None
    engine._ingestion_namespace = None
    engine._drawing_method = None
    engine._drawing_ingestion = None
    engine._analyzed_method = None
    return engine


async def test_successful_query_emits_lifecycle_and_writes_row() -> None:
    obs_sink = RecordingSink()
    tel_sink = TelRecordingSink()
    engine = _build_engine(obs_sink=obs_sink, tel_sink=tel_sink)

    await engine.query("hello", knowledge_id="k1")

    kinds = [r.kind for r in obs_sink.records]
    assert "query.start" in kinds
    assert "query.success" in kinds
    assert len(tel_sink.rows) == 1
    row = tel_sink.rows[0]
    assert isinstance(row, QueryTelemetryRow)
    assert row.outcome == "success"
    assert row.knowledge_id == "k1"
    assert row.grounding_decision == "grounded"
    assert row.duration_ms >= 0
    assert row.query_id


async def test_ungrounded_query_marks_outcome_refused() -> None:
    obs_sink = RecordingSink()
    tel_sink = TelRecordingSink()
    result = _query_result(grounded=False, clarification=None)
    engine = _build_engine(obs_sink=obs_sink, tel_sink=tel_sink, query_result=result)

    await engine.query("hello")

    kinds = [r.kind for r in obs_sink.records]
    assert "query.refused" in kinds
    assert tel_sink.rows[0].outcome == "refused"
    assert tel_sink.rows[0].grounding_decision == "ungrounded"


async def test_clarification_query_marks_outcome_refused() -> None:
    obs_sink = RecordingSink()
    tel_sink = TelRecordingSink()
    result = _query_result(grounded=False, clarification=Clarification(question="?"))
    engine = _build_engine(obs_sink=obs_sink, tel_sink=tel_sink, query_result=result)

    await engine.query("hello")

    assert tel_sink.rows[0].grounding_decision == "clarification"
    assert tel_sink.rows[0].outcome == "refused"


async def test_query_failure_emits_error_and_persists_row() -> None:
    obs_sink = RecordingSink()
    tel_sink = TelRecordingSink()
    engine = _build_engine(obs_sink=obs_sink, tel_sink=tel_sink, raise_exc=RuntimeError("boom"))

    try:
        await engine.query("hello")
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError")

    kinds = [r.kind for r in obs_sink.records]
    assert "query.error" in kinds
    err = next(r for r in obs_sink.records if r.kind == "query.error")
    assert err.error_type == "RuntimeError"
    assert err.error_message == "boom"
    assert err.traceback
    row = tel_sink.rows[0]
    assert row.outcome == "error"
    assert row.error_type == "RuntimeError"
    assert row.error_message == "boom"


async def test_successful_ingest_text_emits_lifecycle_and_writes_row() -> None:
    obs_sink = RecordingSink()
    tel_sink = TelRecordingSink()
    engine = _build_engine(obs_sink=obs_sink, tel_sink=tel_sink)

    source = Source(
        source_id="s-42",
        knowledge_id="k1",
        source_type="text",
        embedding_model="x",
        chunk_count=3,
    )
    fake_service = SimpleNamespace(
        ingest_text=AsyncMock(return_value=source),
    )
    engine._get_ingestion = lambda collection: fake_service  # type: ignore[method-assign]

    await engine.ingest_text("body", knowledge_id="k1", source_type="text")

    kinds = [r.kind for r in obs_sink.records]
    assert "ingest.start" in kinds
    assert "ingest.success" in kinds
    assert len(tel_sink.rows) == 1
    row = tel_sink.rows[0]
    assert isinstance(row, IngestTelemetryRow)
    assert row.outcome == "success"
    assert row.source_id == "s-42"
    assert row.knowledge_id == "k1"
    assert row.chunks_count == 3


async def test_ingest_with_notes_marks_outcome_partial() -> None:
    obs_sink = RecordingSink()
    tel_sink = TelRecordingSink()
    engine = _build_engine(obs_sink=obs_sink, tel_sink=tel_sink)

    source = Source(
        source_id="s-1",
        knowledge_id="k1",
        source_type="text",
        embedding_model="x",
        metadata={"ingestion_notes": ["graph:warn:x", "vision:warn:y"]},
    )
    fake_service = SimpleNamespace(ingest_text=AsyncMock(return_value=source))
    engine._get_ingestion = lambda collection: fake_service  # type: ignore[method-assign]

    await engine.ingest_text("body", knowledge_id="k1")

    assert tel_sink.rows[0].outcome == "partial"
    assert tel_sink.rows[0].notes_count == 2
    kinds = [r.kind for r in obs_sink.records]
    assert "ingest.partial" in kinds


async def test_ingest_failure_emits_error_and_writes_row() -> None:
    obs_sink = RecordingSink()
    tel_sink = TelRecordingSink()
    engine = _build_engine(obs_sink=obs_sink, tel_sink=tel_sink)

    fake_service = SimpleNamespace(ingest_text=AsyncMock(side_effect=ValueError("nope")))
    engine._get_ingestion = lambda collection: fake_service  # type: ignore[method-assign]

    try:
        await engine.ingest_text("body")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")

    kinds = [r.kind for r in obs_sink.records]
    assert "ingest.error" in kinds
    row = tel_sink.rows[0]
    assert row.outcome == "error"
    assert row.error_type == "ValueError"


async def test_query_and_ingest_get_distinct_uuids() -> None:
    obs_sink = RecordingSink()
    tel_sink = TelRecordingSink()
    engine = _build_engine(obs_sink=obs_sink, tel_sink=tel_sink)
    source = Source(source_id="s-x", embedding_model="x")
    fake_service = SimpleNamespace(ingest_text=AsyncMock(return_value=source))
    engine._get_ingestion = lambda collection: fake_service  # type: ignore[method-assign]

    await engine.query("a")
    await engine.ingest_text("b")

    ids: list[Any] = [r.query_id for r in tel_sink.rows if isinstance(r, QueryTelemetryRow)]
    ids += [r.ingest_id for r in tel_sink.rows if isinstance(r, IngestTelemetryRow)]
    assert len(set(ids)) == 2


async def test_explicit_indexed_mode_emits_routing_decision() -> None:
    obs_sink = RecordingSink()
    tel_sink = TelRecordingSink()
    engine = _build_engine(
        obs_sink=obs_sink,
        tel_sink=tel_sink,
        routing=RoutingConfig(mode=QueryMode.INDEXED),
    )

    await engine.query("hello")

    decisions = [r for r in obs_sink.records if r.kind == "routing.decision"]
    assert len(decisions) == 1
    assert decisions[0].context["mode"] == "indexed"
    assert decisions[0].context["reason"] == "explicit_mode"
    assert decisions[0].context["corpus_tokens"] is None
    assert tel_sink.rows[0].routing_decision == "indexed"


async def test_explicit_full_context_mode_emits_routing_decision() -> None:
    obs_sink = RecordingSink()
    tel_sink = TelRecordingSink()
    engine = _build_engine(
        obs_sink=obs_sink,
        tel_sink=tel_sink,
        routing=RoutingConfig(mode=QueryMode.FULL_CONTEXT),
    )
    engine._knowledge_manager = AsyncMock()
    engine._knowledge_manager.get_corpus_tokens = AsyncMock(return_value=42)
    engine._load_full_corpus = AsyncMock(return_value="corpus")  # type: ignore[method-assign]
    engine._generation_service.generate_from_corpus = AsyncMock(return_value=_query_result())

    await engine.query("hello")

    decisions = [r for r in obs_sink.records if r.kind == "routing.decision"]
    assert len(decisions) == 1
    assert decisions[0].context["mode"] == "full_context"
    assert decisions[0].context["reason"] == "explicit_mode"
    assert tel_sink.rows[0].routing_decision == "full_context"


async def test_auto_mode_emits_routing_decision_event_full_context() -> None:
    obs_sink = RecordingSink()
    tel_sink = TelRecordingSink()
    engine = _build_engine(
        obs_sink=obs_sink,
        tel_sink=tel_sink,
        routing=RoutingConfig(mode=QueryMode.AUTO, full_context_threshold=10_000),
    )
    engine._knowledge_manager = AsyncMock()
    engine._knowledge_manager.get_corpus_tokens = AsyncMock(return_value=500)
    engine._load_full_corpus = AsyncMock(return_value="corpus")  # type: ignore[method-assign]
    engine._generation_service.generate_from_corpus = AsyncMock(return_value=_query_result())

    await engine.query("hello")

    decisions = [r for r in obs_sink.records if r.kind == "routing.decision"]
    assert len(decisions) == 1
    assert decisions[0].context["mode"] == "full_context"
    assert decisions[0].context["reason"] == "auto_dispatch"
    assert decisions[0].context["corpus_tokens"] == 500
    assert decisions[0].context["threshold"] == 10_000
    row = tel_sink.rows[0]
    assert row.routing_decision == "full_context"
    assert row.mode == "full_context"
    assert row.corpus_tokens == 500


async def test_auto_mode_emits_routing_decision_event_indexed() -> None:
    obs_sink = RecordingSink()
    tel_sink = TelRecordingSink()
    engine = _build_engine(
        obs_sink=obs_sink,
        tel_sink=tel_sink,
        routing=RoutingConfig(mode=QueryMode.AUTO, full_context_threshold=10_000),
    )
    engine._knowledge_manager = AsyncMock()
    engine._knowledge_manager.get_corpus_tokens = AsyncMock(return_value=200_000)

    await engine.query("hello")

    decisions = [r for r in obs_sink.records if r.kind == "routing.decision"]
    assert len(decisions) == 1
    assert decisions[0].context["mode"] == "indexed"
    assert decisions[0].context["reason"] == "auto_dispatch"
    assert decisions[0].context["corpus_tokens"] == 200_000
    row = tel_sink.rows[0]
    assert row.routing_decision == "indexed"
    assert row.mode == "indexed"
    assert row.corpus_tokens == 200_000


async def test_record_skip_emits_enrichment_skipped_event() -> None:
    from rfnry_rag.ingestion.notes import record_skip
    from rfnry_rag.observability.context import _reset_obs, _set_obs

    obs_sink = RecordingSink()
    obs_token = _set_obs(Observability(sink=obs_sink))
    notes: list[str] = []
    try:
        await record_skip(
            notes,
            step="contextual_chunk",
            level="info",
            reason="document_too_large(500000>134000_cap)",
        )
    finally:
        _reset_obs(obs_token)

    assert notes == ["contextual_chunk:info:document_too_large(500000>134000_cap)"]
    events = [r for r in obs_sink.records if r.kind == "enrichment.skipped"]
    assert len(events) == 1
    assert events[0].context == {
        "step": "contextual_chunk",
        "reason": "document_too_large(500000>134000_cap)",
    }


async def test_record_skip_emits_vision_page_skipped_when_page_number_set() -> None:
    from rfnry_rag.ingestion.notes import record_skip
    from rfnry_rag.observability.context import _reset_obs, _set_obs

    obs_sink = RecordingSink()
    obs_token = _set_obs(Observability(sink=obs_sink))
    notes: list[str] = []
    try:
        await record_skip(
            notes,
            step="vision",
            level="warn",
            reason="page_3:RuntimeError(rate_limited)",
            page_number=3,
        )
    finally:
        _reset_obs(obs_token)

    assert "vision:warn:page_3:RuntimeError(rate_limited)" in notes
    events = [r for r in obs_sink.records if r.kind == "vision.page.skipped"]
    assert len(events) == 1
    assert events[0].context["page_number"] == 3
    assert events[0].context["step"] == "vision"


async def test_record_skip_no_op_outside_obs_context() -> None:
    from rfnry_rag.ingestion.notes import record_skip

    notes: list[str] = []
    await record_skip(notes, step="x", level="info", reason="y")
    assert notes == ["x:info:y"]


async def test_telemetry_persists_via_metadata_store(tmp_path) -> None:
    from rfnry_rag.stores.metadata.sqlalchemy import SQLAlchemyMetadataStore
    from rfnry_rag.telemetry import SqlAlchemyTelemetrySink

    store = SQLAlchemyMetadataStore(f"sqlite:///{tmp_path}/m.db")
    await store.initialize()
    try:
        obs_sink = RecordingSink()
        tel_sink = SqlAlchemyTelemetrySink(metadata_store=store)
        engine = _build_engine(obs_sink=obs_sink, tel_sink=tel_sink)

        await engine.query("hello", knowledge_id="kx")
        rows = await store.list_query_telemetry()
        assert len(rows) == 1
        assert rows[0].knowledge_id == "kx"
    finally:
        await store.shutdown()
