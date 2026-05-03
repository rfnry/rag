from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from rfnry_knowledge.exceptions import ConfigurationError
from rfnry_knowledge.providers import ProviderClient
from rfnry_knowledge.providers.protocols import TokenCounter

if TYPE_CHECKING:
    from rfnry_knowledge.ingestion.base import BaseIngestionMethod, PhasedIngestionMethod


@dataclass
class DocumentExpansionConfig:
    """Opt-in document expansion at index time.

    When enabled, each chunk gets ``num_queries`` LLM-generated synthetic
    questions appended to its embedding/BM25 text — a docT5query-style
    expansion that bridges the user-vocabulary-vs-document-vocabulary gap.
    BEIR shows this beats vanilla BM25 on 11/18 datasets while preserving
    BM25's generalization.

    Defaults are disabled and ``provider_client`` is None — consumers must opt in.
    """

    enabled: bool = False
    num_queries: int = 5
    provider_client: ProviderClient | None = None
    concurrency: int = 5

    def __post_init__(self) -> None:
        if not (1 <= self.num_queries <= 20):
            raise ConfigurationError(f"DocumentExpansionConfig.num_queries={self.num_queries} out of range [1, 20]")
        if not (1 <= self.concurrency <= 100):
            raise ConfigurationError(f"DocumentExpansionConfig.concurrency={self.concurrency} out of range [1, 100]")
        if self.enabled and self.provider_client is None:
            raise ConfigurationError(
                "DocumentExpansionConfig.enabled=True requires provider_client — provide a ProviderClient "
                "(no opinionated default model; consumer chooses)."
            )


@dataclass
class ContextualChunkConfig:
    """Opt-in LLM-generated situating context per chunk at index time.

    Each chunk gets a 50–100 token blob explaining its role in the source
    document, prepended before embedding/BM25.

    Defaults are disabled and ``provider_client`` is None — consumers must opt
    in. Routes through BAML's SituateChunk function.
    """

    enabled: bool = False
    provider_client: ProviderClient | None = None
    token_counter: TokenCounter | None = None
    concurrency: int = 5
    max_context_tokens: int = 100

    def __post_init__(self) -> None:
        if not (1 <= self.concurrency <= 100):
            raise ConfigurationError(f"ContextualChunkConfig.concurrency={self.concurrency} out of range [1, 100]")
        if not (10 <= self.max_context_tokens <= 500):
            raise ConfigurationError(
                f"ContextualChunkConfig.max_context_tokens={self.max_context_tokens} out of range [10, 500]"
            )
        if self.enabled and self.provider_client is None:
            raise ConfigurationError(
                "ContextualChunkConfig.enabled=True requires provider_client — provide a ProviderClient "
                "(no opinionated default model; consumer chooses)."
            )
        if self.enabled and self.token_counter is None:
            raise ConfigurationError(
                "ContextualChunkConfig.enabled=True requires token_counter — supply a TokenCounter "
                "implementation (rfnry-knowledge ships none)."
            )


@dataclass
class IngestionConfig:
    methods: list[BaseIngestionMethod | PhasedIngestionMethod] = field(default_factory=list)
    chunk_size: int = 375
    chunk_overlap: int = 40
    chunk_size_unit: Literal["chars", "tokens"] = "tokens"
    parent_chunk_size: int = -1
    parent_chunk_overlap: int = 200
    chunk_context_headers: bool = True
    document_expansion: DocumentExpansionConfig = field(default_factory=lambda: DocumentExpansionConfig())
    contextual_chunk: ContextualChunkConfig = field(default_factory=lambda: ContextualChunkConfig())
    token_counter: TokenCounter | None = None

    def __post_init__(self) -> None:
        if self.chunk_size_unit not in ("chars", "tokens"):
            raise ConfigurationError(
                f"IngestionConfig.chunk_size_unit must be 'chars' or 'tokens', got {self.chunk_size_unit!r}"
            )
        if self.chunk_size < 1:
            raise ConfigurationError("chunk_size must be positive")
        if self.chunk_overlap < 0:
            raise ConfigurationError("chunk_overlap must be non-negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ConfigurationError("chunk_overlap must be less than chunk_size")
        if self.parent_chunk_size < -1:
            raise ConfigurationError(
                f"parent_chunk_size must be >= -1 (-1=auto, 0=disabled, >0=explicit), got {self.parent_chunk_size}"
            )
        if self.parent_chunk_size == -1:
            self.parent_chunk_size = 3 * self.chunk_size
        if self.parent_chunk_size > 0 and self.parent_chunk_size <= self.chunk_size:
            raise ConfigurationError("parent_chunk_size must be greater than chunk_size")
        if self.parent_chunk_size > 0:
            if self.parent_chunk_overlap < 0:
                raise ConfigurationError("parent_chunk_overlap must be non-negative")
            if self.parent_chunk_overlap >= self.parent_chunk_size:
                raise ConfigurationError(
                    f"parent_chunk_overlap ({self.parent_chunk_overlap}) must be less than "
                    f"parent_chunk_size ({self.parent_chunk_size})"
                )
        # Note: chunk_size_unit='tokens' + token_counter is None is checked at chunker
        # construction time (SemanticChunker.__init__), not here, so configs that never
        # reach the chunker (e.g., test fixtures, drawing-only pipelines) don't trip it.
