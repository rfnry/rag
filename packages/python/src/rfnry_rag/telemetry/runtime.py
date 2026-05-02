from __future__ import annotations

import contextlib

from pydantic import BaseModel

from rfnry_rag.telemetry.record import IngestTelemetryRow, QueryTelemetryRow
from rfnry_rag.telemetry.sink import NullTelemetrySink, TelemetrySink

TelemetryRow = QueryTelemetryRow | IngestTelemetryRow


class Telemetry(BaseModel):
    sink: TelemetrySink = NullTelemetrySink()

    model_config = {"arbitrary_types_allowed": True}

    async def write(self, row: TelemetryRow) -> None:
        with contextlib.suppress(Exception):
            await self.sink.write(row)
