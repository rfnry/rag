from typing import Any
from unittest.mock import AsyncMock

from rfnry_knowledge.telemetry import (
    MemoryAddTelemetryRow,
    MemoryDeleteTelemetryRow,
    MemorySearchTelemetryRow,
    MemoryUpdateTelemetryRow,
    SqlAlchemyTelemetrySink,
)


def test_memory_add_row_minimal() -> None:
    row = MemoryAddTelemetryRow(memory_id="u", outcome="success")
    assert row.row_count == 0
    assert row.dropped_dedup_count == 0


def test_memory_search_row_records_top_score() -> None:
    row = MemorySearchTelemetryRow(
        memory_id="u", outcome="success", result_count=3, top_score=0.81,
    )
    assert row.result_count == 3
    assert row.top_score == 0.81


def test_memory_update_row_carries_before_after_text() -> None:
    row = MemoryUpdateTelemetryRow(
        memory_id="u", memory_row_id="r1", outcome="success",
        text_before="old", text_after="new",
    )
    assert row.text_before == "old"
    assert row.text_after == "new"


def test_memory_delete_row_carries_before_text() -> None:
    row = MemoryDeleteTelemetryRow(
        memory_id="u", memory_row_id="r1", outcome="success", text_before="old",
    )
    assert row.text_before == "old"


async def test_sqlalchemy_sink_dispatches_each_memory_row_type() -> None:
    store: Any = AsyncMock()
    sink = SqlAlchemyTelemetrySink(metadata_store=store)

    await sink.write(MemoryAddTelemetryRow(memory_id="u", outcome="success"))
    store.insert_memory_add_telemetry.assert_awaited_once()

    await sink.write(MemorySearchTelemetryRow(memory_id="u", outcome="success"))
    store.insert_memory_search_telemetry.assert_awaited_once()

    await sink.write(MemoryUpdateTelemetryRow(memory_id="u", memory_row_id="r1", outcome="success"))
    store.insert_memory_update_telemetry.assert_awaited_once()

    await sink.write(MemoryDeleteTelemetryRow(memory_id="u", memory_row_id="r1", outcome="success"))
    store.insert_memory_delete_telemetry.assert_awaited_once()
