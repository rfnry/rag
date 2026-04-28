from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from rfnry_rag.retrieval.baml.baml_client.async_client import b
from rfnry_rag.retrieval.common.errors import ConfigurationError, InputError
from rfnry_rag.retrieval.common.formatting import ChunkOrdering, chunks_to_context
from rfnry_rag.retrieval.common.hashing import file_hash as compute_file_hash

if TYPE_CHECKING:
    from rfnry_rag.retrieval.modules.ingestion.tree.service import TreeIndexingService
    from rfnry_rag.retrieval.modules.retrieval.tree.service import TreeSearchService
from rfnry_rag.retrieval.common.language_model import LanguageModelClient, build_registry
from rfnry_rag.retrieval.common.logging import get_logger
from rfnry_rag.retrieval.common.models import RetrievalTrace, RetrievedChunk, Source
from rfnry_rag.retrieval.modules.evaluation.benchmark import (
    BenchmarkCase,
    BenchmarkConfig,
    BenchmarkReport,
    run_benchmark,
)
from rfnry_rag.retrieval.modules.evaluation.metrics import LLMJudgment
from rfnry_rag.retrieval.modules.generation.models import QueryResult, StepResult, StreamEvent
from rfnry_rag.retrieval.modules.generation.service import GenerationService
from rfnry_rag.retrieval.modules.generation.step import StepGenerationService
from rfnry_rag.retrieval.modules.ingestion.analyze.service import AnalyzedIngestionService
from rfnry_rag.retrieval.modules.ingestion.base import BaseIngestionMethod
from rfnry_rag.retrieval.modules.ingestion.chunk.chunker import SemanticChunker
from rfnry_rag.retrieval.modules.ingestion.chunk.service import IngestionService
from rfnry_rag.retrieval.modules.ingestion.drawing.config import DrawingIngestionConfig
from rfnry_rag.retrieval.modules.ingestion.drawing.service import DrawingIngestionService
from rfnry_rag.retrieval.modules.ingestion.embeddings.base import BaseEmbeddings
from rfnry_rag.retrieval.modules.ingestion.embeddings.sparse.base import BaseSparseEmbeddings
from rfnry_rag.retrieval.modules.ingestion.graph.config import GraphIngestionConfig
from rfnry_rag.retrieval.modules.ingestion.methods.document import DocumentIngestion
from rfnry_rag.retrieval.modules.ingestion.methods.graph import GraphIngestion
from rfnry_rag.retrieval.modules.ingestion.methods.tree import TreeIngestion
from rfnry_rag.retrieval.modules.ingestion.methods.vector import VectorIngestion
from rfnry_rag.retrieval.modules.ingestion.vision.base import BaseVision
from rfnry_rag.retrieval.modules.knowledge.manager import KnowledgeManager
from rfnry_rag.retrieval.modules.knowledge.migration import check_embedding_migration
from rfnry_rag.retrieval.modules.namespace import MethodNamespace
from rfnry_rag.retrieval.modules.retrieval.base import BaseRetrievalMethod
from rfnry_rag.retrieval.modules.retrieval.enrich.service import StructuredRetrievalService
from rfnry_rag.retrieval.modules.retrieval.methods.document import DocumentRetrieval
from rfnry_rag.retrieval.modules.retrieval.methods.graph import GraphRetrieval
from rfnry_rag.retrieval.modules.retrieval.methods.vector import VectorRetrieval
from rfnry_rag.retrieval.modules.retrieval.refinement.base import BaseChunkRefinement
from rfnry_rag.retrieval.modules.retrieval.search.reranking.base import BaseReranking
from rfnry_rag.retrieval.modules.retrieval.search.rewriting.base import BaseQueryRewriting
from rfnry_rag.retrieval.modules.retrieval.search.service import RetrievalService
from rfnry_rag.retrieval.stores.document.base import BaseDocumentStore
from rfnry_rag.retrieval.stores.graph.base import BaseGraphStore
from rfnry_rag.retrieval.stores.metadata.base import BaseMetadataStore
from rfnry_rag.retrieval.stores.vector.base import BaseVectorStore

logger = get_logger("server")

SUPPORTED_STRUCTURED_EXTENSIONS = {".xml", ".l5x"}
SUPPORTED_DRAWING_EXTENSIONS = {".dxf"}  # .pdf is tiebroken via source_type="drawing"

# Size guards on user-supplied inputs. These prevent monetary-DoS (huge query
# sent to every embedding provider) and OOM (unbounded text or metadata). The
# values are conservative but configurable via construction arguments.
_MAX_QUERY_CHARS = 32_000
_MAX_INGEST_CHARS = 5_000_000
_MAX_METADATA_KEYS = 50
_MAX_METADATA_VALUE_CHARS = 8_000


def _validate_query_text(text: str) -> None:
    if len(text) > _MAX_QUERY_CHARS:
        raise InputError(f"query exceeds {_MAX_QUERY_CHARS} chars (got {len(text)})")


def _validate_ingest_content(content: str) -> None:
    if len(content) > _MAX_INGEST_CHARS:
        raise InputError(f"ingest content exceeds {_MAX_INGEST_CHARS} chars (got {len(content)})")


def _validate_metadata(metadata: dict[str, Any] | None) -> None:
    if not metadata:
        return
    if len(metadata) > _MAX_METADATA_KEYS:
        raise InputError(f"metadata exceeds {_MAX_METADATA_KEYS} keys (got {len(metadata)})")
    for k, v in metadata.items():
        if isinstance(v, str) and len(v) > _MAX_METADATA_VALUE_CHARS:
            raise InputError(f"metadata[{k!r}] value exceeds {_MAX_METADATA_VALUE_CHARS} chars (got {len(v)})")


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Use only the provided context to answer questions. "
    "Cite sources with page numbers when available. If the context does not contain "
    "enough information to answer, say so."
)


@dataclass
class PersistenceConfig:
    """Storage backends for the RAG engine.

    Two distinct routing concepts coexist in the engine:

    - **collection** (e.g. ``vector_store.collections=["knowledge", "logs"]``):
      the backend routing key — a Qdrant collection name, filesystem subdir, or
      Postgres schema. Chosen per-ingest/retrieve call via the ``collection=``
      argument and maps 1:1 to a pipeline instance.
    - **knowledge_id** (e.g. ``knowledge_id="tenant-42"``): a per-document
      partition filter applied at query time. Multiple knowledge_ids share the
      same collection; retrieval filters to the requested one.

    Use ``collection`` to physically separate data (different Qdrant clusters
    or schemas); use ``knowledge_id`` to logically partition within one
    collection.
    """

    vector_store: BaseVectorStore | None = None
    metadata_store: BaseMetadataStore | None = None
    document_store: BaseDocumentStore | None = None
    graph_store: BaseGraphStore | None = None


@dataclass
class DocumentExpansionConfig:
    """Opt-in document expansion at index time (R3).

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
            raise ConfigurationError(
                f"DocumentExpansionConfig.num_queries={self.num_queries} out of range [1, 20]"
            )
        if not (1 <= self.concurrency <= 100):
            raise ConfigurationError(
                f"DocumentExpansionConfig.concurrency={self.concurrency} out of range [1, 100]"
            )
        if self.enabled and self.lm_client is None:
            raise ConfigurationError(
                "DocumentExpansionConfig.enabled=True requires lm_client — provide a LanguageModelClient "
                "(no opinionated default model; consumer chooses)."
            )


@dataclass
class IngestionConfig:
    embeddings: BaseEmbeddings | None = None
    vision: BaseVision | None = None
    # 375 tokens per chunk (~1500 chars) — fills modern 512-token embedding windows well.
    # chunk_size is now in token units when chunk_size_unit='tokens' (the default).
    chunk_size: int = 375
    # ~10% overlap preserves cross-boundary context without blowing up chunk count.
    chunk_overlap: int = 40
    # Unit for chunk_size and chunk_overlap: 'tokens' (default) or 'chars'.
    # 'tokens' uses tiktoken cl100k_base; falls back to word count if tiktoken unavailable.
    chunk_size_unit: Literal["chars", "tokens"] = "tokens"
    dpi: int = 300
    lm_client: LanguageModelClient | None = None
    sparse_embeddings: BaseSparseEmbeddings | None = None
    parent_chunk_size: int = -1  # sentinel: -1 means "auto = 3 * chunk_size"; 0 explicitly disables
    parent_chunk_overlap: int = 200
    # Whether to prepend a short source/type header string to each chunk before
    # embedding. This is pure string templating — NOT the LLM-generated
    # contextual-chunking technique the old name implied.
    chunk_context_headers: bool = True
    # Deprecated — use chunk_context_headers. Accepted for one release with a
    # DeprecationWarning; will be removed next major version.
    contextual_chunking: bool | None = None
    # Text-density pre-filter: pages with >= this many extractable chars AND no embedded
    # images skip the vision LLM call and are built from raw text directly. Set to 0 to
    # disable the pre-filter and send every page through vision.
    analyze_text_skip_threshold_chars: int = 300
    # Max concurrent vision LLM calls during PDF analysis. Default 5 stays under
    # Tier-1 rate limits; Tier-2+ accounts can raise this to 8-16 for wall-clock wins.
    analyze_concurrency: int = 5
    # Optional nested config for the DrawingIngestion pipeline (Phase C).
    # None by default so existing consumers are unaffected.
    drawings: DrawingIngestionConfig | None = None
    # Optional nested config for the analyze-path graph mapper (Phase D).
    # None by default so existing consumers are unaffected; when provided,
    # ``GraphIngestionConfig.__post_init__`` handles its own validation.
    graph: GraphIngestionConfig | None = None
    # Opt-in LLM-driven synthetic-query expansion at index time (R3).
    # ``DocumentExpansionConfig`` defaults to disabled; consumers must set
    # ``enabled=True`` AND provide ``lm_client`` to activate.
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
        # Resolve sentinel: -1 means "auto = 3 * chunk_size". Values < -1 are invalid.
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
        # dpi upper bound: beyond ~600 the PDF-to-image buffer grows pathologically
        # (each page can exceed 100MB), causing OOM rather than slow rendering.
        if not (72 <= self.dpi <= 600):
            raise ConfigurationError(f"dpi must be between 72 and 600, got {self.dpi}")
        if not (0 <= self.analyze_text_skip_threshold_chars <= 100_000):
            raise ConfigurationError(
                f"analyze_text_skip_threshold_chars={self.analyze_text_skip_threshold_chars} "
                "out of range [0, 100_000]"
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


@dataclass
class AdaptiveRetrievalConfig:
    """Adaptive retrieval (R5) — query-aware top_k, weights, and expansion.

    R5.1 ships this as plumbing only: `enabled=False` is the default and
    `RetrievalService.retrieve` does not yet read these fields. R5.2
    consumes `top_k_min`/`top_k_max` and `task_weight_profiles` for
    per-query adaptation; R5.3 consumes `confidence_expansion` and
    `max_expansion_retries` for confidence-driven re-retrieval.

    The cross-config rule "`enabled=True AND use_llm_classification=True`
    requires `RetrievalConfig.enrich_lm_client`" lives in
    `RagEngine._validate_config` (cross-config invariants live there by
    convention — see `tree_indexing.max_tokens_per_node`).
    """

    enabled: bool = False
    top_k_min: int = 3
    top_k_max: int = 15
    use_llm_classification: bool = False
    # unbounded: per-method weights are consumer overrides; the fusion stage
    # normalises them at consume time, so no upper bound is meaningful here.
    task_weight_profiles: dict[str, dict[str, float]] | None = None  # unbounded: see comment above
    confidence_expansion: bool = False
    max_expansion_retries: int = 2

    def __post_init__(self) -> None:
        if not (1 <= self.top_k_min <= 50):
            raise ConfigurationError(
                f"AdaptiveRetrievalConfig.top_k_min={self.top_k_min} out of range [1, 50]"
            )
        if not (1 <= self.top_k_max <= 200):
            raise ConfigurationError(
                f"AdaptiveRetrievalConfig.top_k_max={self.top_k_max} out of range [1, 200]"
            )
        if self.top_k_min > self.top_k_max:
            raise ConfigurationError(
                f"AdaptiveRetrievalConfig.top_k_min={self.top_k_min} cannot exceed "
                f"top_k_max={self.top_k_max}"
            )
        if not (0 <= self.max_expansion_retries <= 5):
            raise ConfigurationError(
                f"AdaptiveRetrievalConfig.max_expansion_retries={self.max_expansion_retries} "
                "out of range [0, 5]"
            )


@dataclass
class RetrievalConfig:
    top_k: int = 5
    reranker: BaseReranking | None = None
    query_rewriter: BaseQueryRewriting | None = None
    bm25_enabled: bool = False
    bm25_max_indexes: int = 16
    bm25_max_chunks: int = 50_000
    bm25_tokenizer: Callable[[str], list[str]] | None = None
    source_type_weights: dict[str, float] | None = None
    cross_reference_enrichment: bool = True
    enrich_lm_client: LanguageModelClient | None = None
    parent_expansion: bool = True
    chunk_refiner: BaseChunkRefinement | None = None
    history_window: int = 3
    adaptive: AdaptiveRetrievalConfig = field(default_factory=AdaptiveRetrievalConfig)

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ConfigurationError("top_k must be positive")
        if self.top_k > 200:
            raise ConfigurationError(
                f"top_k must be <= 200, got {self.top_k} — requesting thousands of results OOMs the reranker"
            )
        if self.bm25_max_chunks > 200_000:
            raise ConfigurationError(
                f"bm25_max_chunks must be <= 200_000, got {self.bm25_max_chunks} — "
                "in-memory BM25 index at that size risks OOM; use sparse_embeddings instead"
            )
        if not (1 <= self.bm25_max_indexes <= 1000):
            raise ConfigurationError(
                f"bm25_max_indexes must be 1-1000, got {self.bm25_max_indexes}"
            )
        if not (1 <= self.history_window <= 20):
            raise ConfigurationError(f"history_window must be 1-20, got {self.history_window}")
        if self.source_type_weights is not None:
            for key, weight in self.source_type_weights.items():
                if not 0 < weight <= 10.0:
                    raise ConfigurationError(
                        f"source_type_weights[{key!r}]={weight} — weight must be in (0, 10]"
                    )


@dataclass
class GenerationConfig:
    lm_client: LanguageModelClient | None = None
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    grounding_enabled: bool = False
    grounding_threshold: float = 0.5
    relevance_gate_enabled: bool = False
    relevance_gate_model: LanguageModelClient | None = None
    guiding_enabled: bool = False
    step_lm_client: LanguageModelClient | None = None
    chunk_ordering: ChunkOrdering = ChunkOrdering.SCORE_DESCENDING

    def __post_init__(self) -> None:
        if self.grounding_threshold < 0 or self.grounding_threshold > 1:
            raise ConfigurationError("grounding_threshold must be between 0 and 1")
        if self.relevance_gate_enabled and not self.grounding_enabled:
            raise ConfigurationError("relevance_gate_enabled requires grounding_enabled")
        if self.relevance_gate_enabled and not self.relevance_gate_model:
            raise ConfigurationError("relevance_gate_enabled requires relevance_gate_model")
        if self.guiding_enabled and not self.relevance_gate_enabled:
            raise ConfigurationError("guiding_enabled requires relevance_gate_enabled")
        # Boundary sanity: threshold 0.0 with grounding enabled accepts every
        # answer (no-op); threshold 1.0 blocks every answer. Either is a
        # misconfiguration, though only 0.0 is outright incoherent.
        if self.grounding_enabled and self.grounding_threshold == 0.0:
            raise ConfigurationError("grounding_enabled=True with grounding_threshold=0.0 is a no-op")
        if self.grounding_enabled and self.lm_client is None:
            raise ConfigurationError("grounding_enabled requires lm_client")


@dataclass
class TreeIndexingConfig:
    """Configuration for tree-based document indexing."""

    enabled: bool = False
    model: LanguageModelClient | None = None
    toc_scan_pages: int = 20
    max_pages_per_node: int = 10
    max_tokens_per_node: int = 20_000
    generate_summaries: bool = True
    generate_description: bool = True

    def __post_init__(self) -> None:
        if self.toc_scan_pages < 1:
            raise ConfigurationError("toc_scan_pages must be positive")
        if self.toc_scan_pages > 500:
            raise ConfigurationError(f"toc_scan_pages must be <= 500, got {self.toc_scan_pages}")
        if self.max_pages_per_node < 1:
            raise ConfigurationError("max_pages_per_node must be positive")
        if self.max_pages_per_node > 200:
            raise ConfigurationError(f"max_pages_per_node must be <= 200, got {self.max_pages_per_node}")
        if self.max_tokens_per_node < 1:
            raise ConfigurationError("max_tokens_per_node must be positive")
        if self.max_tokens_per_node > 200_000:
            raise ConfigurationError(f"max_tokens_per_node must be <= 200_000, got {self.max_tokens_per_node}")


@dataclass
class TreeSearchConfig:
    """Configuration for tree-based search."""

    enabled: bool = False
    model: LanguageModelClient | None = None
    max_steps: int = 5
    max_context_tokens: int = 50_000
    max_sources_per_query: int = 50

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ConfigurationError("max_steps must be positive")
        if self.max_steps > 50:
            raise ConfigurationError(f"max_steps must be <= 50, got {self.max_steps}")
        if self.max_context_tokens < 1:
            raise ConfigurationError("max_context_tokens must be positive")
        if self.max_context_tokens > 500_000:
            raise ConfigurationError(f"max_context_tokens must be <= 500_000, got {self.max_context_tokens}")
        if not (1 <= self.max_sources_per_query <= 1000):
            raise ConfigurationError(
                f"max_sources_per_query must be 1-1000, got {self.max_sources_per_query}"
            )


class QueryMode(Enum):
    """User-facing routing strategy chosen per `RagEngine` instance.

    All four modes — `RETRIEVAL` (default, backward-compat), `DIRECT`,
    `HYBRID`, and `AUTO` — are live as of R1.4. The string values match
    the `RetrievalTrace.routing_decision` enumeration.
    """

    RETRIEVAL = "retrieval"
    DIRECT = "direct"
    HYBRID = "hybrid"
    AUTO = "auto"


@dataclass
class RoutingConfig:
    """Top-level routing strategy.

    R1.2 introduces this as the dispatch knob between RETRIEVAL (existing
    pipeline) and DIRECT (full corpus into the prompt; no retrieval). R1.3
    lights up HYBRID — RAG-then-answerability-check (SELF-ROUTE) — and
    enforces `hybrid_answerability_model` as required for that mode. R1.4
    lights up AUTO — per-query routing between RETRIEVAL and DIRECT based
    on `direct_context_threshold` versus the corpus token count.
    """

    mode: QueryMode = QueryMode.RETRIEVAL
    direct_context_threshold: int = 150_000
    hybrid_answerability_model: LanguageModelClient | None = None

    def __post_init__(self) -> None:
        if not (1_000 <= self.direct_context_threshold <= 2_000_000):
            raise ConfigurationError(
                f"RoutingConfig.direct_context_threshold={self.direct_context_threshold} "
                "out of range [1_000, 2_000_000]"
            )
        if self.mode == QueryMode.HYBRID and self.hybrid_answerability_model is None:
            raise ConfigurationError(
                "RoutingConfig.mode=HYBRID requires hybrid_answerability_model"
            )


@dataclass
class RagServerConfig:
    persistence: PersistenceConfig
    ingestion: IngestionConfig
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    tree_indexing: TreeIndexingConfig = field(default_factory=TreeIndexingConfig)
    tree_search: TreeSearchConfig = field(default_factory=TreeSearchConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)


def _derive_embedding_model_name(embeddings: BaseEmbeddings) -> str:
    """Build fingerprint string from provider class and model name."""
    cls_name = type(embeddings).__name__.lower().replace("embeddings", "")
    return f"{cls_name}:{embeddings.model}"


class RagEngine:
    @classmethod
    def vector_only(
        cls,
        *,
        vector_store: BaseVectorStore,
        embeddings: BaseEmbeddings,
        top_k: int = 5,
        reranker: BaseReranking | None = None,
        query_rewriter: BaseQueryRewriting | None = None,
        sparse_embeddings: BaseSparseEmbeddings | None = None,
    ) -> RagServerConfig:
        """Preset: dense vector search only. Add reranker/rewriter for quality."""
        return RagServerConfig(
            persistence=PersistenceConfig(vector_store=vector_store),
            ingestion=IngestionConfig(embeddings=embeddings, sparse_embeddings=sparse_embeddings),
            retrieval=RetrievalConfig(top_k=top_k, reranker=reranker, query_rewriter=query_rewriter),
        )

    @classmethod
    def document_only(
        cls,
        *,
        document_store: BaseDocumentStore,
        top_k: int = 5,
        reranker: BaseReranking | None = None,
    ) -> RagServerConfig:
        """Preset: full-text / substring search only. No embeddings needed."""
        return RagServerConfig(
            persistence=PersistenceConfig(document_store=document_store),
            ingestion=IngestionConfig(),
            retrieval=RetrievalConfig(top_k=top_k, reranker=reranker),
        )

    @classmethod
    def hybrid(
        cls,
        *,
        vector_store: BaseVectorStore,
        embeddings: BaseEmbeddings,
        document_store: BaseDocumentStore | None = None,
        graph_store: BaseGraphStore | None = None,
        sparse_embeddings: BaseSparseEmbeddings | None = None,
        reranker: BaseReranking | None = None,
        query_rewriter: BaseQueryRewriting | None = None,
        top_k: int = 5,
    ) -> RagServerConfig:
        """Preset: multi-path retrieval with optional document/graph/sparse paths + rerank."""
        return RagServerConfig(
            persistence=PersistenceConfig(
                vector_store=vector_store,
                document_store=document_store,
                graph_store=graph_store,
            ),
            ingestion=IngestionConfig(embeddings=embeddings, sparse_embeddings=sparse_embeddings),
            retrieval=RetrievalConfig(top_k=top_k, reranker=reranker, query_rewriter=query_rewriter),
        )

    def __init__(self, config: RagServerConfig) -> None:
        self._config = config
        self._initialized = False
        self._stores_opened = False  # set True before first store.initialize(); guards re-entrant shutdown

        self._ingestion_service: IngestionService | None = None
        self._structured_ingestion: AnalyzedIngestionService | None = None
        self._drawing_ingestion: DrawingIngestionService | None = None
        self._retrieval_service: RetrievalService | None = None
        self._structured_retrieval: StructuredRetrievalService | None = None
        self._generation_service: GenerationService | None = None
        self._knowledge_manager: KnowledgeManager | None = None
        self._step_service: StepGenerationService | None = None
        self._tree_indexing_service: TreeIndexingService | None = None
        self._tree_search_service: TreeSearchService | None = None

        self._retrieval_namespace: MethodNamespace[BaseRetrievalMethod] | None = None
        self._ingestion_namespace: MethodNamespace[BaseIngestionMethod] | None = None

        self._retrieval_by_collection: dict[str, tuple[RetrievalService, StructuredRetrievalService | None]] = {}
        self._ingestion_by_collection: dict[str, IngestionService] = {}

        self._chunker: SemanticChunker | None = None
        self._embedding_model_name: str = ""
        self._expansion_registry: Any = None  # Built in _initialize_impl when document_expansion is enabled
        self._answerability_registry: Any = None  # Built in _initialize_impl when mode=HYBRID

    @property
    def knowledge(self) -> KnowledgeManager:
        self._check_initialized()
        assert self._knowledge_manager is not None
        return self._knowledge_manager

    @property
    def collections(self) -> list[str]:
        """Available Qdrant collections."""
        store = self._config.persistence.vector_store
        if store and hasattr(store, "collections"):
            return store.collections  # type: ignore[no-any-return]
        return []

    @property
    def retrieval(self) -> MethodNamespace[BaseRetrievalMethod]:
        """Namespace of configured retrieval methods."""
        self._check_initialized()
        assert self._retrieval_namespace is not None
        return self._retrieval_namespace

    @property
    def ingestion(self) -> MethodNamespace[BaseIngestionMethod]:
        """Namespace of configured ingestion methods."""
        self._check_initialized()
        assert self._ingestion_namespace is not None
        return self._ingestion_namespace

    def _validate_config(self) -> None:
        """Cross-config validation: ensure at least one retrieval path and required deps."""
        cfg = self._config
        p = cfg.persistence
        i = cfg.ingestion

        has_vector = p.vector_store is not None and i.embeddings is not None
        has_document = p.document_store is not None
        has_graph = p.graph_store is not None

        if not any([has_vector, has_document, has_graph]):
            raise ConfigurationError(
                "At least one retrieval path must be configured: "
                "vector (vector_store + embeddings), "
                "document (document_store), or graph (graph_store)"
            )

        if p.vector_store and not i.embeddings:
            raise ConfigurationError("vector_store requires embeddings")
        if i.embeddings and not p.vector_store:
            raise ConfigurationError("embeddings requires vector_store")

        if has_graph and not i.lm_client:
            # Previously raised at init-time which blocked retrieval-only users
            # with a pre-populated graph. GraphIngestion itself handles the
            # missing-client case at runtime (warn + skip), so degrade this to
            # a one-shot warning here rather than a hard failure.
            logger.warning(
                "graph_store configured without ingestion.lm_client — entity "
                "extraction during ingestion will be skipped. Graph retrieval "
                "still works if the graph is pre-populated."
            )

        if cfg.tree_indexing.enabled and not p.metadata_store:
            raise ConfigurationError("tree_indexing requires metadata_store")
        if cfg.tree_search.enabled and not p.metadata_store:
            raise ConfigurationError("tree_search requires metadata_store")
        if cfg.tree_indexing.enabled and cfg.tree_indexing.model is None:
            raise ConfigurationError("tree_indexing.enabled requires tree_indexing.model")
        if cfg.tree_search.enabled and cfg.tree_search.model is None:
            raise ConfigurationError("tree_search.enabled requires tree_search.model")
        if (
            cfg.tree_indexing.enabled
            and cfg.tree_search.enabled
            and cfg.tree_indexing.max_tokens_per_node > cfg.tree_search.max_context_tokens
        ):
            raise ConfigurationError(
                "tree_indexing.max_tokens_per_node cannot exceed tree_search.max_context_tokens "
                "(a single indexed node would not fit in the search context window)"
            )

        if cfg.retrieval.bm25_enabled and i.sparse_embeddings:
            raise ConfigurationError(
                "bm25_enabled cannot be used together with sparse_embeddings — "
                "sparse embeddings supersede BM25. Disable one."
            )

        if (
            cfg.retrieval.adaptive.enabled
            and cfg.retrieval.adaptive.use_llm_classification
            and cfg.retrieval.enrich_lm_client is None
        ):
            raise ConfigurationError(
                "AdaptiveRetrievalConfig.use_llm_classification=True requires "
                "RetrievalConfig.enrich_lm_client — provide a LanguageModelClient "
                "(no opinionated default model; consumer chooses)."
            )

    async def initialize(self) -> None:
        """Wire all modules and check embedding model consistency.

        On partial failure, already-opened stores are torn down via
        ``shutdown()`` before the exception re-raises. This is needed because
        ``__aexit__`` does not fire when ``__aenter__`` raises, so users
        relying on ``async with RagEngine(...) as rag:`` would otherwise
        leak connections on init failure.
        """
        self._validate_config()
        try:
            await self._initialize_impl()
        except BaseException:
            logger.exception("ragengine init failed — rolling back opened stores")
            try:
                await self.shutdown()
            except Exception:
                logger.exception("error during init-rollback shutdown (continuing)")
            raise

    async def _initialize_impl(self) -> None:
        cfg = self._config
        persistence = cfg.persistence
        ingestion = cfg.ingestion
        retrieval = cfg.retrieval
        gen = cfg.generation

        logger.info("ragengine initializing")

        # Initialize stores
        self._stores_opened = True  # from this point forward, shutdown() must run teardown
        if persistence.metadata_store:
            await persistence.metadata_store.initialize()
        if persistence.document_store:
            await persistence.document_store.initialize()
        if persistence.graph_store:
            await persistence.graph_store.initialize()

        ingestion_methods: list[Any] = []
        retrieval_methods: list[Any] = []

        # Vector path
        if persistence.vector_store and ingestion.embeddings:
            vector_size = await ingestion.embeddings.embedding_dimension()
            await persistence.vector_store.initialize(vector_size)

            self._embedding_model_name = _derive_embedding_model_name(ingestion.embeddings)

            ingestion_methods.append(
                VectorIngestion(
                    vector_store=persistence.vector_store,
                    embeddings=ingestion.embeddings,
                    embedding_model_name=self._embedding_model_name,
                    sparse_embeddings=ingestion.sparse_embeddings,
                    include_synthetic_in_embeddings=ingestion.document_expansion.include_in_embeddings,
                    include_synthetic_in_bm25=ingestion.document_expansion.include_in_bm25,
                )
            )
            retrieval_methods.append(
                VectorRetrieval(
                    vector_store=persistence.vector_store,
                    embeddings=ingestion.embeddings,
                    sparse_embeddings=ingestion.sparse_embeddings,
                    parent_expansion=retrieval.parent_expansion,
                    bm25_enabled=retrieval.bm25_enabled,
                    bm25_max_indexes=retrieval.bm25_max_indexes,
                    bm25_max_chunks=retrieval.bm25_max_chunks,
                    bm25_tokenizer=retrieval.bm25_tokenizer,
                    weight=1.0,
                )
            )

        # Document path
        if persistence.document_store:
            ingestion_methods.append(DocumentIngestion(document_store=persistence.document_store))
            retrieval_methods.append(DocumentRetrieval(document_store=persistence.document_store, weight=0.8))

        # Graph path
        if persistence.graph_store:
            if ingestion.lm_client:
                ingestion_methods.append(
                    GraphIngestion(
                        graph_store=persistence.graph_store,
                        lm_client=ingestion.lm_client,
                        graph_config=ingestion.graph,
                    )
                )
            retrieval_methods.append(GraphRetrieval(graph_store=persistence.graph_store, weight=0.7))

        # Chunker
        self._chunker = SemanticChunker(
            chunk_size=ingestion.chunk_size,
            chunk_overlap=ingestion.chunk_overlap,
            parent_chunk_size=ingestion.parent_chunk_size,
            parent_chunk_overlap=ingestion.parent_chunk_overlap,
            chunk_size_unit=ingestion.chunk_size_unit,
        )

        # Tree indexing service
        self._tree_indexing_service = None
        if cfg.tree_indexing.enabled and persistence.metadata_store:
            from rfnry_rag.retrieval.modules.ingestion.tree.service import TreeIndexingService

            tree_idx_registry = build_registry(cfg.tree_indexing.model) if cfg.tree_indexing.model else None
            self._tree_indexing_service = TreeIndexingService(
                config=cfg.tree_indexing,
                metadata_store=persistence.metadata_store,
                registry=tree_idx_registry,
            )
            ingestion_methods.append(TreeIngestion(tree_service=self._tree_indexing_service))
            logger.info("tree indexing: enabled")

        # Tree search service (requires metadata store for loading tree indexes)
        self._tree_search_service = None
        if cfg.tree_search.enabled and persistence.metadata_store:
            from rfnry_rag.retrieval.modules.retrieval.tree.service import TreeSearchService

            tree_search_registry = build_registry(cfg.tree_search.model) if cfg.tree_search.model else None
            self._tree_search_service = TreeSearchService(
                config=cfg.tree_search,
                registry=tree_search_registry,
            )
            logger.info("tree search: enabled")

        # Build namespaces (public API)
        self._retrieval_namespace = MethodNamespace(retrieval_methods)
        self._ingestion_namespace = MethodNamespace(ingestion_methods)

        # Build expansion registry once (shared across all collection-scoped services)
        # Registry construction is cheap but should not be repeated per-ingest.
        self._expansion_registry = (
            build_registry(ingestion.document_expansion.lm_client)
            if ingestion.document_expansion.enabled and ingestion.document_expansion.lm_client
            else None
        )

        if cfg.routing.mode == QueryMode.HYBRID and cfg.routing.hybrid_answerability_model is not None:
            self._answerability_registry = build_registry(cfg.routing.hybrid_answerability_model)

        # Build services
        self._ingestion_service = IngestionService(
            chunker=self._chunker,
            ingestion_methods=ingestion_methods,
            embedding_model_name=self._embedding_model_name,
            source_type_weights=retrieval.source_type_weights,
            metadata_store=persistence.metadata_store,
            on_ingestion_complete=self._on_ingestion_complete,
            vision_parser=ingestion.vision,
            chunk_context_headers=ingestion.chunk_context_headers,
            document_expansion=ingestion.document_expansion,
            expansion_registry=self._expansion_registry,
        )

        # Analyzed ingestion — shares document method from main list, graph store passed directly
        if persistence.metadata_store and persistence.vector_store and ingestion.embeddings:
            analyzed_methods = [m for m in ingestion_methods if isinstance(m, DocumentIngestion)]
            if not analyzed_methods:
                logger.warning(
                    "structured ingestion enabled but no DocumentIngestion configured — "
                    "analyzed phase 3 will skip document storage"
                )

            self._structured_ingestion = AnalyzedIngestionService(
                embeddings=ingestion.embeddings,
                vector_store=persistence.vector_store,
                metadata_store=persistence.metadata_store,
                embedding_model_name=self._embedding_model_name,
                vision=ingestion.vision,
                dpi=ingestion.dpi,
                source_type_weights=retrieval.source_type_weights,
                on_ingestion_complete=self._on_ingestion_complete,
                lm_client=ingestion.lm_client,
                graph_store=persistence.graph_store,
                ingestion_methods=analyzed_methods,
                analyze_text_skip_threshold_chars=ingestion.analyze_text_skip_threshold_chars,
                analyze_concurrency=ingestion.analyze_concurrency,
                graph_config=ingestion.graph,
            )
            if not ingestion.vision:
                logger.warning("no vision provider — structured PDF analysis disabled")

        # Drawing ingestion — sibling of structured. Enabled only if the consumer
        # explicitly configures IngestionConfig.drawings (opt-in).
        if ingestion.drawings is not None and ingestion.drawings.enabled:
            if persistence.metadata_store is None or persistence.vector_store is None:
                raise ConfigurationError(
                    "DrawingIngestionConfig.enabled=True requires metadata_store and vector_store"
                )
            if ingestion.embeddings is None:
                raise ConfigurationError(
                    "DrawingIngestionConfig.enabled=True requires IngestionConfig.embeddings"
                )
            self._drawing_ingestion = DrawingIngestionService(
                config=ingestion.drawings,
                embeddings=ingestion.embeddings,
                vector_store=persistence.vector_store,
                metadata_store=persistence.metadata_store,
                embedding_model_name=self._embedding_model_name,
                graph_store=persistence.graph_store,
                ingestion_methods=list(ingestion_methods),
            )
            logger.info("drawing ingestion: enabled")

        self._retrieval_service = RetrievalService(
            retrieval_methods=retrieval_methods,
            reranking=retrieval.reranker,
            top_k=retrieval.top_k,
            source_type_weights=retrieval.source_type_weights,
            query_rewriter=retrieval.query_rewriter,
            chunk_refiner=retrieval.chunk_refiner,
        )

        # Structured retrieval (unchanged)
        if persistence.vector_store and ingestion.embeddings:
            self._structured_retrieval = StructuredRetrievalService(
                vector_store=persistence.vector_store,
                embeddings=ingestion.embeddings,
                lm_client=retrieval.enrich_lm_client,
                top_k=retrieval.top_k,
                enrich_cross_references=retrieval.cross_reference_enrichment,
            )

        # Collection-scoped pipelines. Populate both maps symmetrically for every
        # collection — including the first (which reuses the default services) —
        # so retrieval/ingestion against a named collection never silently falls
        # back to the default pipeline.
        self._retrieval_by_collection.clear()
        self._ingestion_by_collection.clear()
        if persistence.vector_store and hasattr(persistence.vector_store, "collections"):
            store_collections: list[str] = persistence.vector_store.collections
            for coll_name in store_collections:
                # INTENTIONAL: the first collection reuses the already-built default service
                # instances. Consequence: retrieving explicitly by the first collection's
                # NAME is equivalent to unscoped retrieval against the default pipeline —
                # they share the same RetrievalService / IngestionService instance (and
                # therefore the same BM25 cache, parent-expansion state, etc.). Later
                # collections get fresh scoped pipelines. Removing this branch requires
                # updating test_default_collection_uses_unscoped_default_services.
                if coll_name == store_collections[0]:
                    self._retrieval_by_collection[coll_name] = (
                        self._retrieval_service,
                        self._structured_retrieval,
                    )
                    assert self._ingestion_service is not None
                    self._ingestion_by_collection[coll_name] = self._ingestion_service
                    continue

                scoped_store = persistence.vector_store.scoped(coll_name)  # type: ignore[attr-defined]
                scoped_retrieval = self._build_retrieval_pipeline(scoped_store, ingestion, retrieval, persistence)
                self._retrieval_by_collection[coll_name] = scoped_retrieval
                self._ingestion_by_collection[coll_name] = self._build_ingestion_service(scoped_store)
                logger.info("pipelines built for collection '%s'", coll_name)

        # Generation
        if gen.lm_client:
            relevance_gate_lm_client = gen.relevance_gate_model if gen.relevance_gate_enabled else None
            self._generation_service = GenerationService(
                lm_client=gen.lm_client,
                system_prompt=gen.system_prompt,
                grounding_enabled=gen.grounding_enabled,
                grounding_threshold=gen.grounding_threshold,
                relevance_gate_enabled=gen.relevance_gate_enabled,
                guiding_enabled=gen.guiding_enabled,
                relevance_gate_lm_client=relevance_gate_lm_client,
                chunk_ordering=gen.chunk_ordering,
            )
            logger.info("generation: enabled")
        else:
            logger.info("generation: disabled (retrieval-only mode)")

        if gen.step_lm_client:
            self._step_service = StepGenerationService(
                lm_client=gen.step_lm_client,
                chunk_ordering=gen.chunk_ordering,
            )
            logger.info("step generation: enabled")

        self._knowledge_manager = KnowledgeManager(
            vector_store=persistence.vector_store,
            metadata_store=persistence.metadata_store,
            on_source_removed=self._on_source_removed,
            document_store=persistence.document_store,
            graph_store=persistence.graph_store,
        )

        stale = await check_embedding_migration(
            metadata_store=persistence.metadata_store,
            embedding_model_name=self._embedding_model_name,
        )
        if stale:
            logger.warning("%d sources are stale and need re-ingestion", stale)

        self._initialized = True

        flows = self._enabled_flows()
        logger.info("ragengine ready — %s flows enabled", ", ".join(flows) if flows else "none")

    async def shutdown(self) -> None:
        """Cleanup all store connections in reverse-init order."""
        if not self._stores_opened:
            return  # idempotent: no stores were opened, or shutdown already ran
        self._stores_opened = False  # prevent re-entrant teardown on a second call
        persistence = self._config.persistence
        # Reverse-init order: vector → graph → document → metadata
        if persistence.vector_store:
            try:
                await persistence.vector_store.shutdown()
            except Exception:
                logger.exception("error shutting down vector store")
        if persistence.graph_store:
            try:
                await persistence.graph_store.shutdown()
            except Exception:
                logger.exception("error shutting down graph store")
        if persistence.document_store:
            try:
                await persistence.document_store.shutdown()
            except Exception:
                logger.exception("error shutting down document store")
        if persistence.metadata_store:
            try:
                await persistence.metadata_store.shutdown()
            except Exception:
                logger.exception("error shutting down metadata store")
        # Null out service refs so post-shutdown access fails cleanly
        self._ingestion_service = None
        self._structured_ingestion = None
        self._drawing_ingestion = None
        self._retrieval_service = None
        self._structured_retrieval = None
        self._generation_service = None
        self._knowledge_manager = None
        self._step_service = None
        self._tree_indexing_service = None
        self._tree_search_service = None
        self._retrieval_namespace = None
        self._ingestion_namespace = None
        self._retrieval_by_collection.clear()
        self._ingestion_by_collection.clear()
        self._initialized = False
        logger.info("ragengine shut down")

    async def __aenter__(self) -> RagEngine:
        await self.initialize()
        return self

    async def __aexit__(self, exc_type: type | None, exc_val: BaseException | None, exc_tb: Any) -> None:
        await self.shutdown()

    async def ingest(
        self,
        file_path: str | Path,
        knowledge_id: str | None = None,
        source_type: str | None = None,
        metadata: dict[str, Any] | None = None,
        page_range: str | None = None,
        resume_from_chunk: int = 0,
        on_progress: Callable[[int, int], Awaitable[None]] | None = None,
        collection: str | None = None,
        tree_index: bool = False,
    ) -> Source:
        """Ingest a file. Routes to unstructured or structured based on extension."""
        self._check_initialized()
        _validate_metadata(metadata)
        file_path = Path(file_path)
        ext = file_path.suffix.lower()

        # Drawing route: .dxf always; .pdf only when source_type='drawing'.
        drawing_route = (
            ext in SUPPORTED_DRAWING_EXTENSIONS
            or (ext == ".pdf" and source_type == "drawing")
        )
        if drawing_route:
            if self._drawing_ingestion is None:
                raise ValueError(
                    "Drawing ingestion not configured. "
                    "Pass IngestionConfig(drawings=DrawingIngestionConfig(enabled=True, ...))."
                )
            if collection is not None:
                raise ValueError(
                    "collection routing is not supported with drawing ingestion"
                )

            # Status-based resume
            metadata_store = self._config.persistence.metadata_store
            existing: Source | None = None
            if metadata_store is not None:
                file_hash_value = await asyncio.to_thread(compute_file_hash, file_path)
                existing = await metadata_store.find_by_hash(file_hash_value, knowledge_id)
            if existing is not None and existing.status == "completed":
                return existing

            if existing is not None and existing.status in {"rendered", "extracted", "linked"}:
                # Resume: skip render, let the stepped calls pick up at the right phase
                source = existing
            else:
                source = await self._drawing_ingestion.render(
                    file_path=file_path,
                    knowledge_id=knowledge_id,
                    source_type=source_type,
                    metadata=metadata,
                )
            if source.status == "rendered":
                source = await self._drawing_ingestion.extract(source.source_id)
            if source.status == "extracted":
                source = await self._drawing_ingestion.link(source.source_id)
            if source.status == "linked":
                source = await self._drawing_ingestion.ingest(source.source_id)
            return source

        if ext in SUPPORTED_STRUCTURED_EXTENSIONS and self._structured_ingestion:
            if collection is not None:
                raise ValueError(
                    f"structured ingestion does not support collection routing "
                    f"(got collection={collection!r}, file type={ext!r})"
                )

            # Status-based resume: find any prior ingest for the same file_hash.
            metadata_store = self._config.persistence.metadata_store
            existing = None
            if metadata_store is not None:
                file_hash_value = await asyncio.to_thread(compute_file_hash, file_path)
                existing = await metadata_store.find_by_hash(file_hash_value, knowledge_id)

            if existing is not None and existing.status == "completed":
                logger.info("[ingest] resume: completed source %s returned", existing.source_id)
                return existing

            if existing is not None and existing.status == "synthesized":
                logger.info("[ingest] resume: continuing from synthesized source %s", existing.source_id)
                return await self._structured_ingestion.ingest(existing.source_id)

            if existing is not None and existing.status == "analyzed":
                logger.info("[ingest] resume: continuing from analyzed source %s", existing.source_id)
                source = await self._structured_ingestion.synthesize(existing.source_id)
                return await self._structured_ingestion.ingest(source.source_id)

            # Fresh run (or no metadata store): all 3 phases
            source = await self._structured_ingestion.analyze(
                file_path=file_path,
                knowledge_id=knowledge_id,
                source_type=source_type,
                metadata=metadata,
                page_range=page_range,
            )
            source = await self._structured_ingestion.synthesize(source.source_id)
            source = await self._structured_ingestion.ingest(source.source_id)
            return source

        ingestion_svc = self._get_ingestion(collection)
        source = await ingestion_svc.ingest(
            file_path=file_path,
            knowledge_id=knowledge_id,
            source_type=source_type,
            metadata=metadata,
            page_range=page_range,
            resume_from_chunk=resume_from_chunk,
            on_progress=on_progress,
        )

        # Tree indexing: build and persist tree index after ingestion
        if tree_index and self._tree_indexing_service:
            from rfnry_rag.retrieval.modules.ingestion.chunk.parsers.pdf import PDFParser
            from rfnry_rag.retrieval.modules.ingestion.tree.toc import PageContent

            # Re-parse the document to get page-level text for tree indexing.
            # Only PDF files support page-level tree indexing currently.
            if ext == ".pdf":
                parser = PDFParser()
                parsed_pages = parser.parse(str(file_path))
                pages = [
                    PageContent(
                        index=p.page_number,
                        text=p.content,
                        token_count=len(p.content) // 4,
                    )
                    for p in parsed_pages
                ]
                doc_name = (metadata or {}).get("name", file_path.name)
                tree_idx = await self._tree_indexing_service.build_tree_index(
                    source_id=source.source_id,
                    doc_name=doc_name,
                    pages=pages,
                )
                await self._tree_indexing_service.save_tree_index(tree_idx)
                logger.info("tree index built for source %s (%d pages)", source.source_id, len(pages))
            else:
                logger.warning("tree_index=True but file type %s does not support tree indexing", ext)

        return source

    async def ingest_text(
        self,
        content: str,
        knowledge_id: str | None = None,
        source_type: str | None = None,
        metadata: dict[str, Any] | None = None,
        collection: str | None = None,
    ) -> Source:
        """Ingest raw text content into the RAG pipeline."""
        self._check_initialized()
        _validate_ingest_content(content)
        _validate_metadata(metadata)
        ingestion_svc = self._get_ingestion(collection)
        return await ingestion_svc.ingest_text(
            content=content, knowledge_id=knowledge_id, source_type=source_type, metadata=metadata
        )

    async def analyze(
        self,
        file_path: str | Path,
        knowledge_id: str | None = None,
        source_type: str | None = None,
        metadata: dict[str, Any] | None = None,
        page_range: str | None = None,
        collection: str | None = None,
    ) -> Source:
        """Structured phase 1: per-page analysis.

        ``collection`` must be ``None`` — structured ingestion does not support
        per-collection routing; a non-None value raises ``ValueError``.
        """
        self._check_initialized()
        if not self._structured_ingestion:
            raise ConfigurationError("metadata store required for structured ingestion")
        if collection is not None:
            raise ValueError(
                f"structured ingestion does not support collection routing (got collection={collection!r})"
            )
        return await self._structured_ingestion.analyze(
            file_path=file_path,
            knowledge_id=knowledge_id,
            source_type=source_type,
            metadata=metadata,
            page_range=page_range,
        )

    async def synthesize(self, source_id: str) -> Source:
        """Structured phase 2: cross-page synthesis."""
        self._check_initialized()
        if not self._structured_ingestion:
            raise ConfigurationError("metadata store required for structured ingestion")
        return await self._structured_ingestion.synthesize(source_id)

    async def complete_ingestion(self, source_id: str) -> Source:
        """Structured phase 3: embed + store."""
        self._check_initialized()
        if not self._structured_ingestion:
            raise ConfigurationError("metadata store required for structured ingestion")
        return await self._structured_ingestion.ingest(source_id)

    async def render_drawing(
        self,
        file_path: str | Path,
        knowledge_id: str | None = None,
        source_type: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Source:
        """Drawing phase 1: render page images."""
        self._check_initialized()
        if self._drawing_ingestion is None:
            raise ConfigurationError(
                "DrawingIngestionConfig not configured — pass IngestionConfig(drawings=...)."
            )
        return await self._drawing_ingestion.render(
            file_path=file_path,
            knowledge_id=knowledge_id,
            source_type=source_type,
            metadata=metadata,
        )

    async def extract_drawing(self, source_id: str) -> Source:
        """Drawing phase 2: per-page DrawingPageAnalysis."""
        self._check_initialized()
        if self._drawing_ingestion is None:
            raise ConfigurationError(
                "DrawingIngestionConfig not configured — pass IngestionConfig(drawings=...)."
            )
        return await self._drawing_ingestion.extract(source_id)

    async def link_drawing(self, source_id: str) -> Source:
        """Drawing phase 3: cross-sheet linking (deterministic + LLM residue)."""
        self._check_initialized()
        if self._drawing_ingestion is None:
            raise ConfigurationError(
                "DrawingIngestionConfig not configured — pass IngestionConfig(drawings=...)."
            )
        return await self._drawing_ingestion.link(source_id)

    async def complete_drawing_ingestion(self, source_id: str) -> Source:
        """Drawing phase 4: embed + graph write."""
        self._check_initialized()
        if self._drawing_ingestion is None:
            raise ConfigurationError(
                "DrawingIngestionConfig not configured — pass IngestionConfig(drawings=...)."
            )
        return await self._drawing_ingestion.ingest(source_id)

    async def query(
        self,
        text: str,
        knowledge_id: str | None = None,
        history: list[tuple[str, str]] | None = None,
        min_score: float | None = None,
        collection: str | None = None,
        system_prompt: str | None = None,
        trace: bool = False,
    ) -> QueryResult:
        """Full pipeline: dispatches on `RoutingConfig.mode`.

        RETRIEVAL (default) runs the existing retrieve-then-generate path.
        DIRECT loads the entire corpus into the prompt and skips retrieval.
        HYBRID runs RAG first, then asks the LLM whether the chunks suffice;
        on "no", escalates to a DIRECT-style full-corpus generation.
        AUTO picks DIRECT or RETRIEVAL per query based on corpus size
        (`RoutingConfig.direct_context_threshold`).
        """
        self._check_initialized()
        _validate_query_text(text)
        if not self._generation_service:
            raise ConfigurationError("query() requires generation.lm_client to be configured")

        mode = self._config.routing.mode
        if mode == QueryMode.RETRIEVAL:
            return await self._query_via_retrieval(
                text, knowledge_id, history, min_score, collection, system_prompt, trace
            )
        if mode == QueryMode.DIRECT:
            return await self._query_via_direct_context(
                text, knowledge_id, history, system_prompt, trace
            )
        if mode == QueryMode.HYBRID:
            return await self._query_via_hybrid(
                text, knowledge_id, history, min_score, collection, system_prompt, trace
            )
        return await self._query_via_auto(
            text, knowledge_id, history, min_score, collection, system_prompt, trace
        )

    async def _query_via_retrieval(
        self,
        text: str,
        knowledge_id: str | None,
        history: list[tuple[str, str]] | None,
        min_score: float | None,
        collection: str | None,
        system_prompt: str | None,
        trace: bool,
    ) -> QueryResult:
        """Existing retrieve-then-generate pipeline (RETRIEVAL mode)."""
        assert self._generation_service is not None
        chunks, trace_obj = await self._retrieve_chunks(
            text, knowledge_id, history, min_score, collection, trace=trace
        )
        grounding_start = time.perf_counter() if trace_obj is not None else 0.0
        result = await self._generation_service.generate(
            query=text, chunks=chunks, history=history, system_prompt=system_prompt
        )
        if trace_obj is not None:
            trace_obj.timings["grounding"] = time.perf_counter() - grounding_start
            trace_obj.routing_decision = "retrieval"
            trace_obj.confidence = result.confidence
            if result.clarification is not None:
                trace_obj.grounding_decision = "clarification"
            elif result.grounded:
                trace_obj.grounding_decision = "grounded"
            else:
                trace_obj.grounding_decision = "ungrounded"
            result.trace = trace_obj
        return result

    async def _query_via_direct_context(
        self,
        text: str,
        knowledge_id: str | None,
        history: list[tuple[str, str]] | None,
        system_prompt: str | None,
        trace: bool,
    ) -> QueryResult:
        """DIRECT: load the entire corpus and answer from full context.

        Skips both grounding and clarification gates: the grounding gate
        scores chunk-level relevance against the query, but DIRECT puts
        the full corpus in the prompt — if the answer isn't there, no
        grounding-retry will fix it. Clarification gates exist for
        ambiguous queries against limited chunks; DIRECT has the entire
        corpus, so the clarification heuristic doesn't apply. Both gates
        would burn LLM calls without changing the outcome.
        """
        assert self._generation_service is not None
        load_start = time.perf_counter() if trace else 0.0
        corpus = await self._load_full_corpus(knowledge_id)
        load_elapsed = time.perf_counter() - load_start if trace else 0.0

        trace_obj: RetrievalTrace | None = None
        if trace:
            trace_obj = RetrievalTrace(
                query=text,
                knowledge_id=knowledge_id,
                routing_decision="direct",
            )
            trace_obj.timings["direct_context_load"] = load_elapsed

        gen_start = time.perf_counter() if trace else 0.0
        result = await self._generation_service.generate_from_corpus(
            query=text, corpus=corpus, history=history, system_prompt=system_prompt
        )
        if trace_obj is not None:
            trace_obj.timings["generation"] = time.perf_counter() - gen_start
            trace_obj.confidence = result.confidence
            result.trace = trace_obj
        return result

    async def _query_via_hybrid(
        self,
        text: str,
        knowledge_id: str | None,
        history: list[tuple[str, str]] | None,
        min_score: float | None,
        collection: str | None,
        system_prompt: str | None,
        trace: bool,
    ) -> QueryResult:
        """HYBRID (SELF-ROUTE): RAG first, then ask the LLM if chunks suffice.

        Phase 1 retrieves chunks via the existing pipeline. Phase 2 calls
        `b.CheckAnswerability` with the chunks-as-context. On `answerable=True`
        we generate from chunks (`routing_decision="hybrid_rag"`); on
        `answerable=False` we load the full corpus and generate from it
        (`routing_decision="hybrid_lc"`).
        """
        assert self._generation_service is not None
        chunks, trace_obj = await self._retrieve_chunks(
            text, knowledge_id, history, min_score, collection, trace=trace
        )
        context = chunks_to_context(chunks, ordering=self._config.generation.chunk_ordering)

        answerable, _reasoning = await self._check_answerability(text, context, trace_obj)

        if answerable:
            gen_start = time.perf_counter() if trace_obj is not None else 0.0
            result = await self._generation_service.generate(
                query=text, chunks=chunks, history=history, system_prompt=system_prompt
            )
            if trace_obj is not None:
                trace_obj.timings["generation"] = time.perf_counter() - gen_start
                # `hybrid_rag` distinguishes "RAG path under HYBRID" from plain
                # `retrieval` so consumers can attribute SELF-ROUTE wins (chunks
                # were sufficient) vs escalations (`hybrid_lc`) downstream.
                trace_obj.routing_decision = "hybrid_rag"
                trace_obj.confidence = result.confidence
                if result.clarification is not None:
                    trace_obj.grounding_decision = "clarification"
                elif result.grounded:
                    trace_obj.grounding_decision = "grounded"
                else:
                    trace_obj.grounding_decision = "ungrounded"
                result.trace = trace_obj
            return result

        load_start = time.perf_counter() if trace_obj is not None else 0.0
        corpus = await self._load_full_corpus(knowledge_id)
        if trace_obj is not None:
            trace_obj.timings["direct_context_load"] = time.perf_counter() - load_start

        gen_start = time.perf_counter() if trace_obj is not None else 0.0
        result = await self._generation_service.generate_from_corpus(
            query=text, corpus=corpus, history=history, system_prompt=system_prompt
        )
        if trace_obj is not None:
            trace_obj.timings["generation"] = time.perf_counter() - gen_start
            # `hybrid_lc` flags an LC escalation (chunks judged insufficient);
            # downstream cost analysis subtracts these from `hybrid_rag` to
            # quantify SELF-ROUTE's expensive-path hit rate.
            trace_obj.routing_decision = "hybrid_lc"
            trace_obj.confidence = result.confidence
            result.trace = trace_obj
        return result

    async def _query_via_auto(
        self,
        text: str,
        knowledge_id: str | None,
        history: list[tuple[str, str]] | None,
        min_score: float | None,
        collection: str | None,
        system_prompt: str | None,
        trace: bool,
    ) -> QueryResult:
        """AUTO: pick DIRECT or RETRIEVAL per query based on corpus token count.

        Reads `KnowledgeManager.get_corpus_tokens` and compares against
        `RoutingConfig.direct_context_threshold`. Below-or-equal routes to
        DIRECT (cheaper-and-better at small sizes); above routes to RETRIEVAL.
        Both delegated paths populate `routing_decision` themselves — AUTO
        adds no new enum value (AUTO is the chosen mode, not the chosen route).
        """
        assert self._knowledge_manager is not None
        tokens = await self._knowledge_manager.get_corpus_tokens(knowledge_id)
        threshold = self._config.routing.direct_context_threshold

        # `tokens <= threshold` (not `<`) so corpora exactly at the threshold
        # take the DIRECT path — DIRECT is the cheaper-and-better strategy
        # for small corpora, and the LLM provider's context window is the
        # actual hard limit, not this routing knob.
        if tokens <= threshold:
            # AUTO does NOT route to HYBRID. HYBRID adds an answerability LLM
            # call to every query; AUTO's job is to pick the cheapest correct
            # strategy. Consumers who want SELF-ROUTE behaviour opt in with
            # `mode="hybrid"` explicitly. See plan §"Why AUTO doesn't route
            # to HYBRID" for the cost-shape argument.
            logger.info(
                "auto routing: tokens=%d threshold=%d → DIRECT",
                tokens,
                threshold,
            )
            return await self._query_via_direct_context(
                text, knowledge_id, history, system_prompt, trace
            )

        logger.info(
            "auto routing: tokens=%d threshold=%d → RETRIEVAL",
            tokens,
            threshold,
        )
        return await self._query_via_retrieval(
            text, knowledge_id, history, min_score, collection, system_prompt, trace
        )

    async def _check_answerability(
        self,
        query: str,
        context: str,
        trace_obj: RetrievalTrace | None,
    ) -> tuple[bool, str]:
        """Run `b.CheckAnswerability` and record timing on the trace.

        Returns `(answerable, reasoning)`. On exception, returns
        `(True, "check_failed: <exc>")` and logs a warning. We treat a
        failed check as answerable=True to degrade to the cheaper RAG path
        rather than silently escalating to LC on a transient error
        (rate limit, timeout, malformed JSON).
        """
        start = time.perf_counter()
        try:
            verdict = await b.CheckAnswerability(
                query=query,
                context=context,
                baml_options={"client_registry": self._answerability_registry},
            )
            if trace_obj is not None:
                trace_obj.timings["answerability_check"] = time.perf_counter() - start
            return bool(verdict.answerable), str(verdict.reasoning)
        except Exception as exc:
            if trace_obj is not None:
                trace_obj.timings["answerability_check"] = time.perf_counter() - start
            logger.warning("answerability check failed; degrading to RAG: %s", exc)
            return True, f"check_failed: {exc}"

    async def query_stream(
        self,
        text: str,
        knowledge_id: str | None = None,
        history: list[tuple[str, str]] | None = None,
        min_score: float | None = None,
        collection: str | None = None,
        system_prompt: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Full pipeline with streaming: retrieval + grounding + streamed LLM generation."""
        self._check_initialized()
        _validate_query_text(text)
        if not self._generation_service:
            raise ConfigurationError("query_stream() requires generation.lm_client to be configured")

        mode = self._config.routing.mode
        if mode != QueryMode.RETRIEVAL:
            raise ConfigurationError(
                f"query_stream() does not support mode={mode.name}; "
                "use query() or set RoutingConfig(mode=RETRIEVAL). "
                "Streaming for non-retrieval modes is deferred."
            )

        chunks, _ = await self._retrieve_chunks(text, knowledge_id, history, min_score, collection)
        async for event in self._generation_service.generate_stream(
            query=text, chunks=chunks, history=history, system_prompt=system_prompt
        ):
            yield event

    async def retrieve(
        self,
        text: str,
        knowledge_id: str | None = None,
        min_score: float | None = None,
        collection: str | None = None,
        trace: bool = False,
    ) -> tuple[list[RetrievedChunk], RetrievalTrace | None]:
        """Low-level retrieval only, no LLM generation.

        Returns ``(chunks, trace)``. With ``trace=False`` (default), the second
        element is ``None`` so callers may unpack ``chunks, _ = await rag.retrieve(...)``
        for chunks-only access. With ``trace=True``, returns the post-refinement
        :class:`RetrievalTrace` with ``grounding_decision`` and ``confidence``
        left as ``None`` (no generation/grounding stage runs in raw retrieval);
        ``final_results`` carries the post-min-score-filter chunks.
        """
        self._check_initialized()
        _validate_query_text(text)
        chunks, trace_obj = await self._retrieve_chunks(
            text, knowledge_id, None, min_score, collection, trace=trace
        )
        return chunks, trace_obj

    async def generate_step(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        context: str | None = None,
    ) -> StepResult:
        """Generate a single reasoning step from retrieved chunks.

        Use with retrieve() to build iterative retrieval loops. The consumer
        owns the loop, stopping conditions, and query enrichment between iterations.
        """
        self._check_initialized()
        _validate_query_text(query)
        if not self._step_service:
            raise ConfigurationError("generate_step() requires generation.step_lm_client to be configured")

        return await self._step_service.generate_step(
            query=query,
            chunks=chunks,
            context=context,
        )

    async def benchmark(
        self,
        cases: list[BenchmarkCase],
        config: BenchmarkConfig | None = None,
        knowledge_id: str | None = None,
        llm_judge: LLMJudgment | None = None,
    ) -> BenchmarkReport:
        """Run cases, collect traces (R8.1) + classifications (R8.2), aggregate.

        Each case is executed via `query(..., trace=True)` so the report
        carries the full per-case trace alongside the aggregate metrics.
        Concurrency is bounded by `config.concurrency` (default 1, serial).
        Pass `llm_judge` only when paying per-case judge LLM cost is
        intended; the judge is `None` in the report when omitted.
        """
        self._check_initialized()
        if not self._generation_service:
            raise ConfigurationError("benchmark() requires generation.lm_client to be configured")

        async def _run_one(query_text: str, *, trace: bool) -> QueryResult:
            return await self.query(query_text, knowledge_id=knowledge_id, trace=trace)

        return await run_benchmark(cases, _run_one, config=config, llm_judge=llm_judge)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts using the configured provider."""
        self._check_initialized()
        if not self._config.ingestion.embeddings:
            raise ConfigurationError("embed() requires embeddings to be configured")
        for text in texts:
            _validate_query_text(text)
        return await self._config.ingestion.embeddings.embed(texts)

    async def embed_single(self, text: str) -> list[float]:
        """Generate an embedding for a single text."""
        self._check_initialized()
        if not self._config.ingestion.embeddings:
            raise ConfigurationError("embed_single() requires embeddings to be configured")
        _validate_query_text(text)
        vectors = await self._config.ingestion.embeddings.embed([text])
        return vectors[0]

    async def _load_full_corpus(self, knowledge_id: str | None) -> str:
        """Concatenate every source's text under ``knowledge_id`` into one string.

        Plumbing for R1's DIRECT / HYBRID modes — they need the whole corpus
        in-prompt, not retrieval-ranked chunks. Caller is responsible for the
        downstream model-context-limit check; this method does not truncate.

        Strategy per source: prefer the document store (lossless, original
        text). Fall back to a vector-store scroll only when the document store
        is absent or returns nothing — the vector path is lossy because chunk
        boundaries can land mid-sentence and table-row chunks are flattened
        into linear text. Skips parent chunks (`chunk_type == "parent"`) on
        the scroll path so Phase A5's parent-child indexing doesn't double
        the emitted text.

        Sequential per-source: a knowledge with N sources issues N reads.
        Acceptable for R1.1 — DIRECT mode invokes this once per query, not
        per chunk. Batch fetch via `IN (?, ?, ...)` is straightforward to add
        if a real workload exposes the latency.

        Returns the empty string when the knowledge has no sources.
        """
        persistence = self._config.persistence
        if persistence.metadata_store is None:
            return ""

        sources = await persistence.metadata_store.list_sources(knowledge_id=knowledge_id)
        # Pin the corpus prefix to a stable function of WHICH sources are
        # present, not WHEN they were ingested. The metadata store today orders
        # by `created_at DESC`, which means re-ingesting a source bumps it to
        # the front and busts the entire prompt cache. Sorting by source_id
        # here makes re-ingestion cache-invalidate only the affected slot, and
        # adding a new source only changes the suffix below existing ones.
        sources = sorted(sources, key=lambda s: s.source_id)
        document_store = persistence.document_store
        vector_store = persistence.vector_store
        parts: list[str] = []
        for source in sources:
            text: str | None = None
            if document_store is not None:
                text = await document_store.get(source.source_id)
            if not text and vector_store is not None:
                text = await self._reconstruct_corpus_from_vector_scroll(
                    vector_store,
                    source.source_id,
                )
            if not text:
                continue
            name = source.metadata.get("name", source.source_id)
            parts.append(f"[Source: {name}]\n{text}")
        return "\n\n".join(parts)

    @staticmethod
    async def _reconstruct_corpus_from_vector_scroll(
        vector_store: BaseVectorStore,
        source_id: str,
    ) -> str:
        """Reassemble text by scrolling child chunks for ``source_id``.

        Lossy fallback — chunk boundaries are not guaranteed to align with
        sentence/paragraph boundaries, and table-row chunks emit one linear
        line per row. Filters parent chunks (`chunk_type == "parent"`) so
        Phase A5's parent-child indexing doesn't double-count text.
        """
        offset: str | None = None
        ordered: list[tuple[int, str]] = []
        while True:
            results, next_offset = await vector_store.scroll(
                filters={"source_id": source_id},
                limit=500,
                offset=offset,
            )
            for r in results:
                if r.payload.get("chunk_type", "child") == "parent":
                    continue
                content = r.payload.get("content", "")
                if not content:
                    continue
                idx = r.payload.get("chunk_index", 0)
                ordered.append((idx, content))
            if next_offset is None or not results:
                break
            offset = next_offset
        ordered.sort(key=lambda pair: pair[0])
        return "\n\n".join(text for _, text in ordered)

    def _build_retrieval_query(self, text: str, history: list[tuple[str, str]] | None) -> str:
        """Enrich the retrieval query with recent conversation context.

        When history is available, appends key terms from recent exchanges so
        retrieval can find results relevant to the ongoing conversation — not
        just the latest message in isolation.
        """
        if not history:
            return text

        recent = history[-self._config.retrieval.history_window :]
        context_parts = []
        for human_msg, _assistant_msg in recent:
            context_parts.append(human_msg)

        context = " ".join(context_parts)
        return f"{text}\n\nConversation context: {context}"

    async def _invalidate_vector_caches(self, knowledge_id: str | None) -> None:
        """Invalidate BM25 cache on every scoped collection's VectorRetrieval, not
        just the default one. Scoped collections carry their own method instances
        with their own caches — iterate them all so stale BM25 results are not
        returned after a cross-collection ingest/remove."""
        seen: set[int] = set()
        if self._retrieval_by_collection:
            # SERIAL: invalidate_cache mutates in-memory BM25 state; the `seen`
            # set prevents double-invalidation of shared method instances across
            # collections. Concurrent invalidations on the same method instance
            # would require locking the BM25 cache — serial is simpler and safe.
            for retrieval_service, _ in self._retrieval_by_collection.values():
                for method in retrieval_service.methods:
                    if method.name != "vector" or id(method) in seen:
                        continue
                    seen.add(id(method))
                    if hasattr(method, "invalidate_cache"):
                        await method.invalidate_cache(knowledge_id)
            return
        # No multi-collection wiring — fall back to the default namespace.
        if self._retrieval_namespace and "vector" in self._retrieval_namespace:
            vector = self._retrieval_namespace.vector
            if hasattr(vector, "invalidate_cache"):
                await vector.invalidate_cache(knowledge_id)

    async def _on_ingestion_complete(self, knowledge_id: str | None) -> None:
        """Callback after ingestion — invalidates BM25 cache for the knowledge_id."""
        await self._invalidate_vector_caches(knowledge_id)

    async def _on_source_removed(self, knowledge_id: str | None) -> None:
        """Callback after source removal — invalidates BM25 cache for the knowledge_id."""
        await self._invalidate_vector_caches(knowledge_id)

    def _build_retrieval_pipeline(
        self,
        vector_store: BaseVectorStore,
        ingestion: IngestionConfig,
        retrieval: RetrievalConfig,
        persistence: PersistenceConfig,
    ) -> tuple[RetrievalService, StructuredRetrievalService | None]:
        methods: list = []
        if ingestion.embeddings:
            methods.append(
                VectorRetrieval(
                    vector_store=vector_store,
                    embeddings=ingestion.embeddings,
                    sparse_embeddings=ingestion.sparse_embeddings,
                    parent_expansion=retrieval.parent_expansion,
                    bm25_enabled=retrieval.bm25_enabled,
                    bm25_max_indexes=retrieval.bm25_max_indexes,
                    bm25_max_chunks=retrieval.bm25_max_chunks,
                    bm25_tokenizer=retrieval.bm25_tokenizer,
                    weight=1.0,
                )
            )
        if persistence.document_store:
            methods.append(DocumentRetrieval(document_store=persistence.document_store, weight=0.8))
        if persistence.graph_store:
            methods.append(GraphRetrieval(graph_store=persistence.graph_store, weight=0.7))

        unstructured = RetrievalService(
            retrieval_methods=methods,
            reranking=retrieval.reranker,
            top_k=retrieval.top_k,
            source_type_weights=retrieval.source_type_weights,
            query_rewriter=retrieval.query_rewriter,
            chunk_refiner=retrieval.chunk_refiner,
        )
        structured: StructuredRetrievalService | None = None
        if ingestion.embeddings:
            structured = StructuredRetrievalService(
                vector_store=vector_store,
                embeddings=ingestion.embeddings,
                lm_client=retrieval.enrich_lm_client,
                top_k=retrieval.top_k,
                enrich_cross_references=retrieval.cross_reference_enrichment,
            )
        return unstructured, structured

    def _build_ingestion_service(self, vector_store: BaseVectorStore) -> IngestionService:
        assert self._chunker is not None
        cfg = self._config
        methods: list = []
        if cfg.ingestion.embeddings:
            methods.append(
                VectorIngestion(
                    vector_store=vector_store,
                    embeddings=cfg.ingestion.embeddings,
                    embedding_model_name=self._embedding_model_name,
                    sparse_embeddings=cfg.ingestion.sparse_embeddings,
                    include_synthetic_in_embeddings=cfg.ingestion.document_expansion.include_in_embeddings,
                    include_synthetic_in_bm25=cfg.ingestion.document_expansion.include_in_bm25,
                )
            )
        if cfg.persistence.document_store:
            methods.append(DocumentIngestion(document_store=cfg.persistence.document_store))
        if cfg.persistence.graph_store and cfg.ingestion.lm_client:
            methods.append(
                GraphIngestion(
                    graph_store=cfg.persistence.graph_store,
                    lm_client=cfg.ingestion.lm_client,
                    graph_config=cfg.ingestion.graph,
                )
            )
        if self._tree_indexing_service is not None:
            methods.append(TreeIngestion(tree_service=self._tree_indexing_service))

        return IngestionService(
            chunker=self._chunker,
            ingestion_methods=methods,
            embedding_model_name=self._embedding_model_name,
            source_type_weights=cfg.retrieval.source_type_weights,
            metadata_store=cfg.persistence.metadata_store,
            on_ingestion_complete=self._on_ingestion_complete,
            vision_parser=cfg.ingestion.vision,
            chunk_context_headers=cfg.ingestion.chunk_context_headers,
            document_expansion=cfg.ingestion.document_expansion,
            expansion_registry=self._expansion_registry,
        )

    async def _retrieve_chunks(
        self,
        text: str,
        knowledge_id: str | None,
        history: list[tuple[str, str]] | None,
        min_score: float | None,
        collection: str | None,
        trace: bool = False,
    ) -> tuple[list[RetrievedChunk], RetrievalTrace | None]:
        """Shared retrieval: unstructured + structured merge + score filter.

        When `trace=True`, returns the unstructured-pipeline trace alongside
        the merged chunks. The trace's `final_results` reflect the
        post-min-score-filter view returned to the caller.
        """
        unstructured, structured = self._get_retrieval(collection)
        retrieval_query = self._build_retrieval_query(text, history)

        tree_chunks: list[RetrievedChunk] = []
        if self._tree_search_service and self._config.persistence.metadata_store:
            tree_chunks = await self._run_tree_search(
                query=retrieval_query,
                knowledge_id=knowledge_id,
            )

        tree_kwargs: dict[str, Any] = {"tree_chunks": tree_chunks} if tree_chunks else {}

        trace_obj: RetrievalTrace | None = None
        if structured:
            # return_exceptions so one path failing doesn't kill the query.
            # The inner search service already degrades per-variant; this makes
            # the structured-vs-unstructured merge consistent.
            results = await asyncio.gather(
                unstructured.retrieve(
                    query=retrieval_query, knowledge_id=knowledge_id, trace=trace, **tree_kwargs
                ),
                structured.retrieve(query=retrieval_query, knowledge_id=knowledge_id),
                return_exceptions=True,
            )
            if isinstance(results[0], BaseException):
                logger.warning("unstructured retrieval failed: %s", results[0])
                unstructured_chunks: list[RetrievedChunk] = []
            else:
                unstructured_chunks, trace_obj = results[0]
            structured_chunks = results[1] if not isinstance(results[1], BaseException) else []
            if isinstance(results[1], BaseException):
                logger.warning("structured retrieval failed: %s", results[1])
            chunks = self._merge_retrieval_results(unstructured_chunks, structured_chunks)  # type: ignore[arg-type]
        else:
            chunks, trace_obj = await unstructured.retrieve(
                query=retrieval_query, knowledge_id=knowledge_id, trace=trace, **tree_kwargs
            )

        if min_score is not None:
            chunks = [c for c in chunks if c.score >= min_score]

        if trace_obj is not None:
            trace_obj.final_results = list(chunks)

        return chunks, trace_obj

    async def _run_tree_search(
        self,
        query: str,
        knowledge_id: str | None,
    ) -> list[RetrievedChunk]:
        """Load tree indexes for relevant sources and run tree search concurrently across sources."""
        import json

        from rfnry_rag.retrieval.common.models import TreeIndex
        from rfnry_rag.retrieval.modules.ingestion.tree.toc import PageContent

        assert self._tree_search_service is not None
        tree_service = self._tree_search_service
        metadata_store = self._config.persistence.metadata_store
        assert metadata_store is not None

        source_ids_full = await metadata_store.list_source_ids(knowledge_id=knowledge_id)

        max_sources = self._config.tree_search.max_sources_per_query
        if len(source_ids_full) > max_sources:
            logger.warning(
                "tree search limited to %d of %d sources (max_sources_per_query)",
                max_sources,
                len(source_ids_full),
            )
            source_ids_full = source_ids_full[:max_sources]

        tree_jsons = await metadata_store.get_tree_indexes(source_ids_full)

        async def search_one(source_id: str) -> list[RetrievedChunk]:
            tree_json = tree_jsons.get(source_id)
            if not tree_json:
                return []
            tree_index = TreeIndex.from_dict(json.loads(tree_json))
            if not tree_index.pages:
                logger.warning("tree index for %s has no stored pages, skipping tree search", source_id)
                return []
            # Convert stored TreePage back to PageContent for the search service
            pages = [PageContent(index=p.index, text=p.text, token_count=p.token_count) for p in tree_index.pages]
            results = await tree_service.search(
                query=query,
                tree_index=tree_index,
                pages=pages,
            )
            if not results:
                return []
            return tree_service.to_retrieved_chunks(results, tree_index)

        per_source = await asyncio.gather(
            *(search_one(sid) for sid in source_ids_full),
            return_exceptions=True,
        )

        all_tree_chunks: list[RetrievedChunk] = []
        for source_id, outcome in zip(source_ids_full, per_source, strict=True):
            if isinstance(outcome, BaseException):
                logger.warning("tree search for %s failed: %s — skipping", source_id, outcome)
                continue
            all_tree_chunks.extend(outcome)
        return all_tree_chunks

    def _get_retrieval(self, collection: str | None) -> tuple[RetrievalService, StructuredRetrievalService | None]:
        """Return retrieval pipeline for *collection* (default if None).

        Raises ValueError when *collection* is specified but unknown — previously
        this silently fell back to the default pipeline, which could mix data
        across collections without any warning.
        """
        if collection is None:
            assert self._retrieval_service is not None
            return self._retrieval_service, self._structured_retrieval
        if collection in self._retrieval_by_collection:
            return self._retrieval_by_collection[collection]
        raise ValueError(f"unknown collection {collection!r}; known: {sorted(self._retrieval_by_collection.keys())}")

    def _get_ingestion(self, collection: str | None) -> IngestionService:
        """Return ingestion service for *collection*.

        Raises ValueError when *collection* is specified but unknown — same
        reasoning as _get_retrieval: silently falling back would mix data
        across collections.
        """
        if collection is None:
            assert self._ingestion_service is not None
            return self._ingestion_service

        if collection in self._ingestion_by_collection:
            return self._ingestion_by_collection[collection]

        raise ValueError(f"unknown collection {collection!r}; known: {sorted(self._ingestion_by_collection.keys())}")

    def _check_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError(
                "RagEngine not initialized. Use 'async with RagEngine(config) as rag:' "
                "or call 'await rag.initialize()' first."
            )

    def _enabled_flows(self) -> list[str]:
        flows: list[str] = []
        if self._retrieval_namespace:
            flows.extend(m.name for m in self._retrieval_namespace)
        if self._structured_ingestion:
            flows.append("structured")
        if self._generation_service:
            flows.append("generation")
        if self._tree_search_service:
            flows.append("tree_search")
        return flows

    @staticmethod
    def _merge_retrieval_results(
        unstructured: list[RetrievedChunk],
        structured: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """Merge unstructured and structured results via reciprocal rank fusion.

        Raw-score sort is unsafe here because unstructured chunks carry RRF scores
        (~0.01-0.05) and structured chunks carry cosine scores (0-1). Sorting by
        the raw value always places structured results first regardless of
        relevance. RRF is scale-free and uses rank position instead of raw score.
        """
        if not structured:
            return unstructured
        if not unstructured:
            return structured
        from rfnry_rag.retrieval.modules.retrieval.search.fusion import reciprocal_rank_fusion

        return reciprocal_rank_fusion([unstructured, structured])
