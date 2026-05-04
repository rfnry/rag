from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from rfnry_knowledge.config.entity import EntityIngestionConfig
from rfnry_knowledge.exceptions import ConfigurationError
from rfnry_knowledge.observability import Observability
from rfnry_knowledge.providers import (
    BaseEmbeddings,
    BaseReranking,
    BaseSparseEmbeddings,
    ProviderClient,
)
from rfnry_knowledge.telemetry import Telemetry

if TYPE_CHECKING:
    from rfnry_knowledge.memory.extraction import BaseExtractor
    from rfnry_knowledge.stores.document.base import BaseDocumentStore
    from rfnry_knowledge.stores.graph.base import BaseGraphStore
    from rfnry_knowledge.stores.metadata.base import BaseMetadataStore
    from rfnry_knowledge.stores.vector.base import BaseVectorStore


@dataclass(frozen=True)
class MemoryIngestionConfig:
    extractor: BaseExtractor
    embeddings: BaseEmbeddings
    sparse_embeddings: BaseSparseEmbeddings | None = None
    entity_extraction: EntityIngestionConfig | None = None
    semantic_required: bool = True
    keyword_required: bool = False
    entity_required: bool = False
    dedup_context_top_k: int = 0
    dedup_context_recent_turns: int = 3
    keyword_backend: Literal["bm25", "postgres_fts"] = "bm25"
    bm25_max_chunks: int = 50_000

    def __post_init__(self) -> None:
        if self.dedup_context_top_k < 0:
            raise ConfigurationError("dedup_context_top_k must be >= 0")
        if self.dedup_context_recent_turns < 1:
            raise ConfigurationError("dedup_context_recent_turns must be >= 1")
        if self.bm25_max_chunks < 1:
            raise ConfigurationError("bm25_max_chunks must be >= 1")
        if self.keyword_backend not in ("bm25", "postgres_fts"):
            raise ConfigurationError(f"unknown keyword_backend {self.keyword_backend!r}")


@dataclass(frozen=True)
class MemoryRetrievalConfig:
    semantic_weight: float = 0.5
    keyword_weight: float = 0.3
    entity_weight: float = 0.2
    entity_hops: int = 2
    rerank: BaseReranking | None = None
    over_fetch_multiplier: int = 4

    def __post_init__(self) -> None:
        for name, w in (
            ("semantic_weight", self.semantic_weight),
            ("keyword_weight", self.keyword_weight),
            ("entity_weight", self.entity_weight),
        ):
            if w < 0:
                raise ConfigurationError(f"{name} must be >= 0, got {w}")
        if self.semantic_weight + self.keyword_weight + self.entity_weight <= 0:
            raise ConfigurationError("at least one of semantic/keyword/entity weight must be > 0")
        if self.entity_hops < 1:
            raise ConfigurationError("entity_hops must be >= 1")
        if self.over_fetch_multiplier < 1:
            raise ConfigurationError("over_fetch_multiplier must be >= 1")


@dataclass
class MemoryEngineConfig:
    ingestion: MemoryIngestionConfig
    retrieval: MemoryRetrievalConfig
    vector_store: BaseVectorStore
    provider: ProviderClient
    document_store: BaseDocumentStore | None = None
    graph_store: BaseGraphStore | None = None
    metadata_store: BaseMetadataStore | None = None
    observability: Observability = field(default_factory=Observability)
    telemetry: Telemetry = field(default_factory=Telemetry)

    def __post_init__(self) -> None:
        if self.ingestion.keyword_backend == "postgres_fts" and self.document_store is None:
            raise ConfigurationError(
                "MemoryEngineConfig.document_store is required when keyword_backend='postgres_fts'"
            )
        if self.ingestion.entity_extraction is not None and self.graph_store is None:
            raise ConfigurationError(
                "MemoryEngineConfig.graph_store is required when entity_extraction is set"
            )
