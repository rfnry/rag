from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from rfnry_rag.exceptions import ConfigurationError
from rfnry_rag.providers import GenerativeModelClient

if TYPE_CHECKING:
    from rfnry_rag.ingestion.base import BaseIngestionMethod, PhasedIngestionMethod


@dataclass
class DocumentExpansionConfig:
    """Opt-in document expansion at index time.

    When enabled, each chunk gets ``num_queries`` LLM-generated synthetic
    questions appended to its embedding/BM25 text — a docT5query-style
    expansion that bridges the user-vocabulary-vs-document-vocabulary gap.
    BEIR shows this beats vanilla BM25 on 11/18 datasets while preserving
    BM25's generalization.

    Defaults are disabled and ``lm_client`` is None — consumers must opt in.
    """

    enabled: bool = False
    num_queries: int = 5
    lm_client: GenerativeModelClient | None = None
    concurrency: int = 5

    def __post_init__(self) -> None:
        if not (1 <= self.num_queries <= 20):
            raise ConfigurationError(f"DocumentExpansionConfig.num_queries={self.num_queries} out of range [1, 20]")
        if not (1 <= self.concurrency <= 100):
            raise ConfigurationError(f"DocumentExpansionConfig.concurrency={self.concurrency} out of range [1, 100]")
        if self.enabled and self.lm_client is None:
            raise ConfigurationError(
                "DocumentExpansionConfig.enabled=True requires lm_client — provide a GenerativeModelClient "
                "(no opinionated default model; consumer chooses)."
            )


@dataclass
class ContextualChunkConfig:
    """Opt-in LLM-generated situating context per chunk at index time.

    Anthropic's "Contextual Retrieval" recipe: each chunk gets a 50–100
    token blob explaining its role in the source document, prepended
    before embedding/BM25. Reduces retrieval failures 35–67% vs
    structural-header-only chunking on Anthropic's reported corpora.

    Defaults are disabled and ``lm_client`` is None — consumers must opt
    in. Native SDK dispatch (no BAML); ``lm_client`` carries credentials,
    retries, timeout, temperature.
    """

    enabled: bool = False
    lm_client: GenerativeModelClient | None = None
    concurrency: int = 5
    max_context_tokens: int = 100

    def __post_init__(self) -> None:
        if not (1 <= self.concurrency <= 100):
            raise ConfigurationError(
                f"ContextualChunkConfig.concurrency={self.concurrency} out of range [1, 100]"
            )
        if not (10 <= self.max_context_tokens <= 500):
            raise ConfigurationError(
                f"ContextualChunkConfig.max_context_tokens={self.max_context_tokens} out of range [10, 500]"
            )
        if self.enabled and self.lm_client is None:
            raise ConfigurationError(
                "ContextualChunkConfig.enabled=True requires lm_client — provide a GenerativeModelClient "
                "(no opinionated default model; consumer chooses)."
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
