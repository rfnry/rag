from __future__ import annotations

from dataclasses import dataclass, field

from rfnry_rag.config.generation import GenerationConfig
from rfnry_rag.config.ingestion import IngestionConfig
from rfnry_rag.config.retrieval import RetrievalConfig
from rfnry_rag.config.routing import RoutingConfig
from rfnry_rag.stores.metadata.base import BaseMetadataStore


@dataclass
class RagEngineConfig:
    """Top-level engine config.

    Stores other than ``metadata_store`` live inside the ingestion / retrieval
    methods that use them — share the same instance across methods to point
    them at the same backend (or pass distinct instances to use different
    ones).

    ``metadata_store`` is the one cross-cutting store used directly by the
    engine + ``KnowledgeManager`` (``corpus_tokens``, ``list``, ``remove``)
    and by every ingestion method to write Source rows. It is not method-
    specific, so it lives at the engine level.
    """

    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    metadata_store: BaseMetadataStore | None = None
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
