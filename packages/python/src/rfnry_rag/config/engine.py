from __future__ import annotations

from dataclasses import dataclass, field

from rfnry_rag.config.generation import GenerationConfig
from rfnry_rag.config.ingestion import IngestionConfig
from rfnry_rag.config.persistence import PersistenceConfig
from rfnry_rag.config.retrieval import RetrievalConfig
from rfnry_rag.config.routing import RoutingConfig


@dataclass
class RagEngineConfig:
    persistence: PersistenceConfig
    ingestion: IngestionConfig
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)


# Back-compat alias — ``RagServerConfig`` was renamed to ``RagEngineConfig``;
# the engine class is ``RagEngine``, so the config name now matches.
RagServerConfig = RagEngineConfig
