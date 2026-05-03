from datetime import datetime
from typing import Protocol

from rfnry_knowledge.models import Source, SourceStats
from rfnry_knowledge.telemetry.record import IngestTelemetryRow, QueryTelemetryRow


class BaseMetadataStore(Protocol):
    async def initialize(self) -> None: ...

    async def create_source(self, source: Source) -> None: ...

    async def get_source(self, source_id: str) -> Source | None: ...

    async def list_sources(self, knowledge_id: str | None = None) -> list[Source]: ...

    async def list_source_ids(self, knowledge_id: str | None = None) -> list[str]: ...

    async def update_source(self, source_id: str, **fields) -> None: ...

    async def delete_source(self, source_id: str) -> None: ...

    async def record_hit(self, source_id: str, chunk_id: str, grounded: bool) -> None: ...

    async def find_by_hash(self, hash_value: str, knowledge_id: str | None) -> Source | None: ...

    async def get_source_stats(self, source_id: str) -> SourceStats | None: ...

    async def save_tree_index(self, source_id: str, tree_index_json: str) -> None: ...

    async def get_tree_index(self, source_id: str) -> str | None: ...

    async def get_tree_indexes(self, source_ids: list[str]) -> dict[str, str | None]: ...

    async def upsert_page_analyses(
        self,
        source_id: str,
        analyses: list[dict],
    ) -> None: ...

    async def get_page_analyses(self, source_id: str) -> list[dict]: ...

    async def get_page_analyses_by_hash(
        self,
        page_hashes: list[str],
        knowledge_id: str | None,
    ) -> dict[str, dict]: ...

    async def get_page_analysis(
        self,
        source_id: str,
        page_number: int,
    ) -> dict | None: ...

    async def insert_query_telemetry(self, row: QueryTelemetryRow) -> None: ...

    async def insert_ingest_telemetry(self, row: IngestTelemetryRow) -> None: ...

    async def list_query_telemetry(
        self,
        *,
        knowledge_id: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 1000,
    ) -> list[QueryTelemetryRow]: ...

    async def list_ingest_telemetry(
        self,
        *,
        knowledge_id: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 1000,
    ) -> list[IngestTelemetryRow]: ...

    async def shutdown(self) -> None: ...
