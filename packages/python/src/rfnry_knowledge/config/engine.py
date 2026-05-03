from __future__ import annotations

from dataclasses import dataclass, field

from rfnry_knowledge.config.generation import GenerationConfig
from rfnry_knowledge.config.ingestion import IngestionConfig
from rfnry_knowledge.config.retrieval import RetrievalConfig
from rfnry_knowledge.config.routing import RoutingConfig
from rfnry_knowledge.observability import Observability
from rfnry_knowledge.stores.metadata.base import BaseMetadataStore
from rfnry_knowledge.telemetry import (
    NullTelemetrySink,
    SqlAlchemyTelemetrySink,
    Telemetry,
)


@dataclass
class KnowledgeEngineConfig:
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    metadata_store: BaseMetadataStore | None = None
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)

    observability: Observability = field(default_factory=Observability)
    telemetry: Telemetry = field(default_factory=Telemetry)

    def __post_init__(self) -> None:
        if isinstance(self.telemetry.sink, NullTelemetrySink) and self.metadata_store is not None:
            self.telemetry = Telemetry(sink=SqlAlchemyTelemetrySink(metadata_store=self.metadata_store))
