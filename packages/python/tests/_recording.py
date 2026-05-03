from __future__ import annotations

from rfnry_knowledge.observability import ObservabilityRecord
from rfnry_knowledge.telemetry import IngestTelemetryRow, QueryTelemetryRow


class RecordingObservabilitySink:
    def __init__(self) -> None:
        self.records: list[ObservabilityRecord] = []

    async def emit(self, record: ObservabilityRecord) -> None:
        self.records.append(record)


class RecordingTelemetrySink:
    def __init__(self) -> None:
        self.rows: list[QueryTelemetryRow | IngestTelemetryRow] = []

    async def write(self, row: QueryTelemetryRow | IngestTelemetryRow) -> None:
        self.rows.append(row)
