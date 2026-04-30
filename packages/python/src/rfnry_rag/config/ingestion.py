from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from rfnry_rag.config.drawing import DrawingIngestionConfig
from rfnry_rag.config.graph import GraphIngestionConfig
from rfnry_rag.exceptions import ConfigurationError
from rfnry_rag.ingestion.embeddings.base import BaseEmbeddings
from rfnry_rag.ingestion.embeddings.sparse.base import BaseSparseEmbeddings
from rfnry_rag.ingestion.vision.base import BaseVision
from rfnry_rag.providers import LanguageModelClient


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
    lm_client: LanguageModelClient | None = None
    include_in_embeddings: bool = True
    include_in_bm25: bool = True
    concurrency: int = 5

    def __post_init__(self) -> None:
        if not (1 <= self.num_queries <= 20):
            raise ConfigurationError(f"DocumentExpansionConfig.num_queries={self.num_queries} out of range [1, 20]")
        if not (1 <= self.concurrency <= 100):
            raise ConfigurationError(f"DocumentExpansionConfig.concurrency={self.concurrency} out of range [1, 100]")
        if self.enabled and self.lm_client is None:
            raise ConfigurationError(
                "DocumentExpansionConfig.enabled=True requires lm_client — provide a LanguageModelClient "
                "(no opinionated default model; consumer chooses)."
            )


@dataclass
class IngestionConfig:
    embeddings: BaseEmbeddings | None = None
    vision: BaseVision | None = None
    chunk_size: int = 375
    chunk_overlap: int = 40
    chunk_size_unit: Literal["chars", "tokens"] = "tokens"
    dpi: int = 300
    lm_client: LanguageModelClient | None = None
    sparse_embeddings: BaseSparseEmbeddings | None = None
    parent_chunk_size: int = -1
    parent_chunk_overlap: int = 200
    chunk_context_headers: bool = True
    contextual_chunking: bool | None = None
    analyze_text_skip_threshold_chars: int = 300
    analyze_concurrency: int = 5
    drawings: DrawingIngestionConfig | None = None
    graph: GraphIngestionConfig | None = None
    document_expansion: DocumentExpansionConfig = field(default_factory=lambda: DocumentExpansionConfig())

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
        if not (72 <= self.dpi <= 600):
            raise ConfigurationError(f"dpi must be between 72 and 600, got {self.dpi}")
        if not (0 <= self.analyze_text_skip_threshold_chars <= 100_000):
            raise ConfigurationError(
                f"analyze_text_skip_threshold_chars={self.analyze_text_skip_threshold_chars} out of range [0, 100_000]"
            )
        if not (1 <= self.analyze_concurrency <= 100):
            raise ConfigurationError(
                f"IngestionConfig.analyze_concurrency={self.analyze_concurrency} out of range [1, 100]"
            )
        if self.contextual_chunking is not None:
            import warnings

            warnings.warn(
                "contextual_chunking is deprecated; use chunk_context_headers. "
                "The old name implied LLM-generated context (Anthropic-style) but "
                "the implementation is pure string templating.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.chunk_context_headers = self.contextual_chunking
