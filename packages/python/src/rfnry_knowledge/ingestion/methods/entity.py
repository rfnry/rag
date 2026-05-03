from __future__ import annotations

import time
from typing import Any

from rfnry_knowledge.common.logging import get_logger
from rfnry_knowledge.config.entity import EntityIngestionConfig
from rfnry_knowledge.ingestion.analyze.models import DiscoveredEntity, PageAnalysis
from rfnry_knowledge.ingestion.models import ChunkedContent, ParsedPage
from rfnry_knowledge.ingestion.notes import record_skip
from rfnry_knowledge.observability.context import current_obs
from rfnry_knowledge.providers import ProviderClient, build_registry
from rfnry_knowledge.stores.graph.base import BaseGraphStore
from rfnry_knowledge.stores.graph.mapper import page_entities_to_graph
from rfnry_knowledge.telemetry.context import current_ingest_row
from rfnry_knowledge.telemetry.usage import instrument_baml_call

logger = get_logger("ingestion.methods.entity")

# Lazy import — avoid circular dependency and heavy BAML import at module level
b: Any = None


def _get_baml_client() -> Any:
    global b
    if b is None:
        from rfnry_knowledge.baml.baml_client.async_client import b as _b

        b = _b
    return b


class EntityIngestion:
    """Extract entities from text via LLM and store in graph store.

    Uses the ``ExtractEntitiesFromText`` BAML function to extract entities,
    then maps them to ``GraphEntity`` via ``page_entities_to_graph()`` and
    stores via ``graph_store.add_entities()``.
    """

    required: bool = False

    def __init__(
        self,
        store: BaseGraphStore,
        provider_client: ProviderClient | None = None,
        graph_config: EntityIngestionConfig | None = None,
    ) -> None:
        self._store = store
        self._provider_client = provider_client
        self._registry = build_registry(provider_client) if provider_client else None
        self._graph_config = graph_config if graph_config is not None else EntityIngestionConfig()

    def clone_for_store(self, store: BaseGraphStore) -> EntityIngestion:
        return EntityIngestion(
            store=store,
            provider_client=self._provider_client,
            graph_config=self._graph_config,
        )

    @property
    def name(self) -> str:
        return "entity"

    async def ingest(
        self,
        source_id: str,
        knowledge_id: str | None,
        source_type: str | None,
        source_weight: float,
        title: str,
        full_text: str,
        chunks: list[ChunkedContent],
        tags: list[str],
        metadata: dict[str, Any],
        hash_value: str | None = None,
        pages: list[ParsedPage] | None = None,
        notes: list[str] | None = None,
    ) -> None:
        if not self._registry:
            logger.warning("graph ingestion skipped — no provider_client provided")
            return

        start = time.perf_counter()
        obs = current_obs()
        ingest_row = current_ingest_row()
        registry = self._registry
        try:
            client = _get_baml_client()
            result = await instrument_baml_call(
                operation="extract_entities",
                call=lambda collector: client.ExtractEntitiesFromText(
                    full_text,
                    baml_options={"client_registry": registry, "collector": collector},
                ),
            )

            if not result.entities:
                elapsed = (time.perf_counter() - start) * 1000
                logger.info("no entities found in %.1fms", elapsed)
                return

            analysis = PageAnalysis(
                page_number=1,
                description=result.description,
                entities=[
                    DiscoveredEntity(
                        name=e.name,
                        category=e.category,
                        value=e.value,
                        context=e.context,
                    )
                    for e in result.entities
                ],
                tables=[],
                annotations=result.annotations if result.annotations else [],
                page_type=result.page_type or "text",
            )

            graph_entities = page_entities_to_graph(analysis, source_id, self._graph_config)

            await self._store.add_entities(
                source_id=source_id,
                knowledge_id=knowledge_id,
                entities=graph_entities,
            )

            elapsed_ms = int((time.perf_counter() - start) * 1000)
            logger.info("%d entities extracted and stored in %dms", len(graph_entities), elapsed_ms)
            if obs is not None:
                await obs.emit(
                    "ingestion.method.success",
                    f"{self.name} ingest ok",
                    source_id=source_id,
                    context={
                        "method_name": self.name,
                        "duration_ms": elapsed_ms,
                        "entities": len(graph_entities),
                    },
                )

        except Exception as exc:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            logger.warning("failed in %dms — %s", elapsed_ms, exc)
            await record_skip(
                notes,
                step="graph",
                level="warn",
                reason=f"extraction_failed({exc!s:.80})",
            )
            if ingest_row is not None:
                ingest_row.graph_extraction_failed = True
            if obs is not None:
                await obs.emit(
                    "ingestion.method.error",
                    f"{self.name} ingest failed",
                    level="error",
                    source_id=source_id,
                    context={"method_name": self.name, "duration_ms": elapsed_ms},
                    error=exc,
                )

    async def delete(self, source_id: str) -> None:
        await self._store.delete_by_source(source_id)
