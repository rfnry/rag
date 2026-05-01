from __future__ import annotations

from dataclasses import dataclass, field

from rfnry_rag.telemetry.record import IngestTelemetryRow, QueryTelemetryRow
from rfnry_rag.telemetry.sinks import JsonlStderrSink, Sink

TelemetryRow = QueryTelemetryRow | IngestTelemetryRow


@dataclass
class Telemetry:
    """Row-per-transaction emitter. Always-on; pass `NullSink()` to silence."""

    sink: Sink = field(default_factory=JsonlStderrSink)

    async def write(self, row: TelemetryRow) -> None:
        await self.sink.write(row)
