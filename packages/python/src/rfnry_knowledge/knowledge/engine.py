from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any

from rfnry_knowledge.common.logging import get_logger
from rfnry_knowledge.config.engine import KnowledgeEngineConfig
from rfnry_knowledge.config.ingestion import IngestionConfig
from rfnry_knowledge.config.retrieval import RetrievalConfig
from rfnry_knowledge.config.routing import QueryMode
from rfnry_knowledge.exceptions import ConfigurationError, InputError
from rfnry_knowledge.generation.models import QueryResult, StreamEvent
from rfnry_knowledge.generation.service import GenerationService
from rfnry_knowledge.ingestion.analyze.service import AnalyzedIngestionService
from rfnry_knowledge.ingestion.base import BaseIngestionMethod
from rfnry_knowledge.ingestion.chunk.chunker import SemanticChunker
from rfnry_knowledge.ingestion.chunk.service import IngestionService
from rfnry_knowledge.ingestion.drawing.service import DrawingIngestionService
from rfnry_knowledge.ingestion.embeddings.base import BaseEmbeddings
from rfnry_knowledge.ingestion.embeddings.batching import embed_batched
from rfnry_knowledge.ingestion.embeddings.sparse.base import BaseSparseEmbeddings
from rfnry_knowledge.ingestion.hashing import file_hash as compute_file_hash
from rfnry_knowledge.ingestion.methods import (
    AnalyzedIngestion,
    DrawingIngestion,
    EntityIngestion,
    KeywordIngestion,
    SemanticIngestion,
)
from rfnry_knowledge.ingestion.vision.base import BaseVision
from rfnry_knowledge.knowledge.manager import KnowledgeManager
from rfnry_knowledge.knowledge.migration import check_embedding_migration
from rfnry_knowledge.models import RetrievedChunk, Source
from rfnry_knowledge.observability import Observability
from rfnry_knowledge.observability.benchmark import (
    BenchmarkCase,
    BenchmarkConfig,
    BenchmarkReport,
    run_benchmark,
)
from rfnry_knowledge.observability.context import _reset_obs, _set_obs
from rfnry_knowledge.observability.metrics import LLMJudgment
from rfnry_knowledge.observability.trace import RetrievalTrace
from rfnry_knowledge.providers import build_registry
from rfnry_knowledge.retrieval.base import BaseRetrievalMethod
from rfnry_knowledge.retrieval.methods.entity import EntityRetrieval
from rfnry_knowledge.retrieval.methods.keyword import KeywordRetrieval
from rfnry_knowledge.retrieval.methods.semantic import SemanticRetrieval
from rfnry_knowledge.retrieval.namespace import MethodNamespace
from rfnry_knowledge.retrieval.search.reranking.base import BaseReranking
from rfnry_knowledge.retrieval.search.service import RetrievalService
from rfnry_knowledge.stores.document.base import BaseDocumentStore
from rfnry_knowledge.stores.graph.base import BaseGraphStore
from rfnry_knowledge.stores.vector.base import BaseVectorStore
from rfnry_knowledge.telemetry import IngestTelemetryRow, QueryTelemetryRow, Telemetry
from rfnry_knowledge.telemetry.context import _reset_row, _set_row

logger = get_logger("engine")

# Size guards on user-supplied inputs. These prevent monetary-DoS (huge query
# sent to every embedding provider) and OOM (unbounded text or metadata). The
# values are conservative but configurable via construction arguments.
_MAX_QUERY_CHARS = 32_000
_MAX_INGEST_CHARS = 5_000_000
_MAX_METADATA_KEYS = 50
_MAX_METADATA_VALUE_CHARS = 8_000

# Reserve subtracted from the generation provider's advertised window when
# validating that ``RoutingConfig.full_context_threshold`` fits in FULL_CONTEXT
# mode. Covers system prompt (~2k) + chat history (~6k) + the public query cap
# (~8k tokens for 32k chars). Output tokens are added separately from
# ``ProviderClient.max_tokens``.
_FULL_CONTEXT_NON_OUTPUT_RESERVE_TOKENS = 16_000


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


def _derive_embedding_model_name(embeddings: BaseEmbeddings) -> str:
    """Build fingerprint string from provider class and model name."""
    cls_name = type(embeddings).__name__.lower().replace("embeddings", "")
    return f"{cls_name}:{embeddings.model}"


class KnowledgeEngine:
    @classmethod
    def vector_only(
        cls,
        *,
        vector_store: BaseVectorStore,
        embeddings: BaseEmbeddings,
        embedding_model_name: str = "",
        metadata_store: Any = None,
        top_k: int = 5,
        reranker: BaseReranking | None = None,
        sparse_embeddings: BaseSparseEmbeddings | None = None,
    ) -> KnowledgeEngineConfig:
        """Preset: dense vector search only."""
        name = embedding_model_name or _derive_embedding_model_name(embeddings)
        return KnowledgeEngineConfig(
            metadata_store=metadata_store,
            ingestion=IngestionConfig(
                methods=[
                    SemanticIngestion(
                        store=vector_store,
                        embeddings=embeddings,
                        embedding_model_name=name,
                        sparse_embeddings=sparse_embeddings,
                    )
                ],
            ),
            retrieval=RetrievalConfig(
                methods=[
                    SemanticRetrieval(store=vector_store, embeddings=embeddings, sparse_embeddings=sparse_embeddings)
                ],
                top_k=top_k,
                reranker=reranker,
            ),
        )

    @classmethod
    def document_only(
        cls,
        *,
        document_store: BaseDocumentStore,
        metadata_store: Any = None,
        top_k: int = 5,
        reranker: BaseReranking | None = None,
    ) -> KnowledgeEngineConfig:
        """Preset: full-text / substring search only. No embeddings needed."""
        return KnowledgeEngineConfig(
            metadata_store=metadata_store,
            ingestion=IngestionConfig(methods=[KeywordIngestion(store=document_store)]),
            retrieval=RetrievalConfig(
                methods=[KeywordRetrieval(backend="postgres_fts", document_store=document_store, weight=0.8)],
                top_k=top_k,
                reranker=reranker,
            ),
        )

    @classmethod
    def hybrid(
        cls,
        *,
        vector_store: BaseVectorStore,
        embeddings: BaseEmbeddings,
        document_store: BaseDocumentStore | None = None,
        graph_store: BaseGraphStore | None = None,
        metadata_store: Any = None,
        sparse_embeddings: BaseSparseEmbeddings | None = None,
        reranker: BaseReranking | None = None,
        top_k: int = 5,
    ) -> KnowledgeEngineConfig:
        """Preset: multi-path retrieval with optional document/graph/sparse paths + rerank."""
        name = _derive_embedding_model_name(embeddings)
        ing_methods: list[Any] = [
            SemanticIngestion(
                store=vector_store,
                embeddings=embeddings,
                embedding_model_name=name,
                sparse_embeddings=sparse_embeddings,
            )
        ]
        ret_methods: list[Any] = [
            SemanticRetrieval(store=vector_store, embeddings=embeddings, sparse_embeddings=sparse_embeddings)
        ]
        if document_store is not None:
            ing_methods.append(KeywordIngestion(store=document_store))
            ret_methods.append(KeywordRetrieval(backend="postgres_fts", document_store=document_store, weight=0.8))
        if graph_store is not None:
            ret_methods.append(EntityRetrieval(store=graph_store, weight=0.7))

        return KnowledgeEngineConfig(
            metadata_store=metadata_store,
            ingestion=IngestionConfig(methods=ing_methods),
            retrieval=RetrievalConfig(methods=ret_methods, top_k=top_k, reranker=reranker),
        )

    def __init__(self, config: KnowledgeEngineConfig) -> None:
        self._config = config
        self._observability: Observability = config.observability
        self._telemetry: Telemetry = config.telemetry
        self._initialized = False
        self._stores_opened = False  # set True before first store.initialize(); guards re-entrant shutdown

        self._ingestion_service: IngestionService | None = None
        self._structured_ingestion: AnalyzedIngestionService | None = None
        self._drawing_ingestion: DrawingIngestionService | None = None
        self._analyzed_method: AnalyzedIngestion | None = None
        self._drawing_method: DrawingIngestion | None = None
        self._retrieval_service: RetrievalService | None = None
        self._generation_service: GenerationService | None = None
        self._knowledge_manager: KnowledgeManager | None = None

        self._retrieval_namespace: MethodNamespace[BaseRetrievalMethod] | None = None
        self._ingestion_namespace: MethodNamespace[BaseIngestionMethod] | None = None

        self._retrieval_by_collection: dict[str, RetrievalService] = {}
        self._ingestion_by_collection: dict[str, IngestionService] = {}

        self._chunker: SemanticChunker | None = None
        self._embedding_model_name: str = ""
        self._expansion_registry: Any = None

        # Stores discovered from configured methods at initialize() time.
        # Populated by _discover_stores(); used by sub-pipelines (drawing,
        # analyzed) and by the lifecycle teardown in shutdown().
        self._vector_store: BaseVectorStore | None = None
        self._document_store: BaseDocumentStore | None = None
        self._graph_store: BaseGraphStore | None = None

    def _discover_stores(self) -> None:
        """Walk configured methods, capture distinct store instances."""
        methods: list[Any] = list(self._config.ingestion.methods) + list(self._config.retrieval.methods)
        for m in methods:
            store = getattr(m, "_store", None)
            if store is None:
                continue
            if isinstance(m, SemanticIngestion | SemanticRetrieval) and self._vector_store is None:
                self._vector_store = store
            elif (
                isinstance(m, KeywordIngestion) or (isinstance(m, KeywordRetrieval) and m.backend == "postgres_fts")
            ) and self._document_store is None:
                self._document_store = store
            elif isinstance(m, EntityIngestion | EntityRetrieval) and self._graph_store is None:
                self._graph_store = store

    @property
    def knowledge(self) -> KnowledgeManager:
        self._check_initialized()
        assert self._knowledge_manager is not None
        return self._knowledge_manager

    @property
    def collections(self) -> list[str]:
        """Available Qdrant collections."""
        store = self._vector_store
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
        """Cross-config validation: ensure at least one retrieval method is configured
        and the FULL_CONTEXT routing threshold fits inside the generation provider's
        advertised window when one is declared."""
        cfg = self._config
        if not cfg.retrieval.methods:
            raise ConfigurationError(
                "RetrievalConfig.methods must not be empty — configure at least one "
                "retrieval method (SemanticRetrieval, KeywordRetrieval, or EntityRetrieval)."
            )
        self._validate_full_context_fits_provider_window()

    def _validate_full_context_fits_provider_window(self) -> None:
        """When the generation provider declares ``context_size``, ensure the
        FULL_CONTEXT path cannot overflow the model's window.

        Reserve = ``_FULL_CONTEXT_NON_OUTPUT_RESERVE_TOKENS`` (system prompt +
        history + question cap) + ``provider_client.max_tokens`` (output budget).
        Skipped when generation has no client or the provider omits
        ``context_size``.
        """
        cfg = self._config
        client = cfg.generation.provider_client
        if client is None:
            return
        window = client.context_size
        if window is None:
            return
        threshold = cfg.routing.full_context_threshold
        output_reserve = client.max_tokens
        total_reserve = _FULL_CONTEXT_NON_OUTPUT_RESERVE_TOKENS + output_reserve
        if threshold + total_reserve > window:
            raise ConfigurationError(
                f"RoutingConfig.full_context_threshold={threshold} + reserve={total_reserve} "
                f"({_FULL_CONTEXT_NON_OUTPUT_RESERVE_TOKENS} non-output + {output_reserve} max_tokens output) "
                f"exceeds {client.name}.context_size={window}. "
                f"Lower full_context_threshold or raise context_size."
            )

    async def initialize(self) -> None:
        """Wire all modules and check embedding model consistency.

        On partial failure, already-opened stores are torn down via
        ``shutdown()`` before the exception re-raises. This is needed because
        ``__aexit__`` does not fire when ``__aenter__`` raises, so users
        relying on ``async with KnowledgeEngine(...) as engine:`` would otherwise
        leak connections on init failure.
        """
        self._validate_config()
        try:
            await self._initialize_impl()
        except BaseException:
            logger.exception("knowledge engine init failed — rolling back opened stores")
            try:
                await self.shutdown()
            except Exception:
                logger.exception("error during init-rollback shutdown (continuing)")
            raise

    async def _initialize_impl(self) -> None:
        cfg = self._config
        ingestion = cfg.ingestion
        retrieval = cfg.retrieval
        gen = cfg.generation
        metadata_store = cfg.metadata_store

        logger.info("knowledge engine initializing")

        # Discover stores from configured methods.
        self._discover_stores()

        # Initialize stores
        self._stores_opened = True  # from this point forward, shutdown() must run teardown
        if metadata_store:
            await metadata_store.initialize()
        if self._document_store:
            await self._document_store.initialize()
        if self._graph_store:
            await self._graph_store.initialize()

        ingestion_methods: list[Any] = list(ingestion.methods)
        retrieval_methods: list[Any] = list(retrieval.methods)

        # standard_methods drives the regular IngestionService chunked dispatch
        # AND serves as delegate_methods for the analyzed pipeline's phase-3
        # fan-out. Excluding the phased wrappers prevents double-running them
        # on plain ingest paths.
        standard_methods: list[Any] = [
            m for m in ingestion_methods if not isinstance(m, AnalyzedIngestion | DrawingIngestion)
        ]

        # Vector store init: walk the methods list for an instance carrying an
        # ``_embeddings`` reference (SemanticIngestion / AnalyzedIngestion /
        # DrawingIngestion all expose it) and ask it for the embedding
        # dimension to pre-create the store collection.
        embeddings_for_dim: BaseEmbeddings | None = next(
            (m._embeddings for m in ingestion_methods if hasattr(m, "_embeddings")),
            None,
        )
        if self._vector_store is not None and embeddings_for_dim is not None:
            vector_size = await embeddings_for_dim.embedding_dimension()
            await self._vector_store.initialize(vector_size)
            self._embedding_model_name = _derive_embedding_model_name(embeddings_for_dim)

        # Chunker
        self._chunker = SemanticChunker(
            chunk_size=ingestion.chunk_size,
            chunk_overlap=ingestion.chunk_overlap,
            parent_chunk_size=ingestion.parent_chunk_size,
            parent_chunk_overlap=ingestion.parent_chunk_overlap,
            chunk_size_unit=ingestion.chunk_size_unit,
            token_counter=ingestion.token_counter,
        )

        # Build namespaces (public API)
        self._retrieval_namespace = MethodNamespace(retrieval_methods)
        self._ingestion_namespace = MethodNamespace(ingestion_methods)

        logger.info(
            "ingestion methods: %s",
            ", ".join(m.name for m in ingestion_methods) or "(none)",
        )
        logger.info(
            "retrieval methods: %s",
            ", ".join(f"{m.name}(weight={m.weight})" for m in retrieval_methods) or "(none)",
        )

        # Build expansion registry once (shared across all collection-scoped services)
        # Registry construction is cheap but should not be repeated per-ingest.
        self._expansion_registry = (
            build_registry(ingestion.document_expansion.provider_client)
            if ingestion.document_expansion.enabled and ingestion.document_expansion.provider_client
            else None
        )

        # Phased pipelines: method-list dispatch.
        analyzed_method = next((m for m in ingestion_methods if isinstance(m, AnalyzedIngestion)), None)
        drawing_method = next((m for m in ingestion_methods if isinstance(m, DrawingIngestion)), None)

        # Build services. ``vision_parser`` is optional and only used by the
        # standard chunker path for image extensions (.jpg/.png/...); pull it
        # from the analyzed/drawing wrapper if one is configured.
        vision_parser: BaseVision | None = None
        if analyzed_method is not None:
            vision_parser = analyzed_method._vision
        elif drawing_method is not None:
            vision_parser = drawing_method._vision

        self._ingestion_service = IngestionService(
            chunker=self._chunker,
            ingestion_methods=standard_methods,
            embedding_model_name=self._embedding_model_name,
            source_type_weights=retrieval.source_type_weights,
            metadata_store=metadata_store,
            on_ingestion_complete=self._on_ingestion_complete,
            vision_parser=vision_parser,
            chunk_context_headers=ingestion.chunk_context_headers,
            document_expansion=ingestion.document_expansion,
            expansion_registry=self._expansion_registry,
            contextual_chunk=ingestion.contextual_chunk,
            token_counter=ingestion.token_counter,
        )

        if analyzed_method is not None and metadata_store is not None:
            document_delegates = [m for m in standard_methods if isinstance(m, KeywordIngestion)]
            if not document_delegates:
                logger.warning(
                    "AnalyzedIngestion configured but no KeywordIngestion in methods — "
                    "phase 3 will skip document storage"
                )
            analyzed_method.bind(
                metadata_store=metadata_store,
                delegate_methods=document_delegates,
                on_ingestion_complete=self._on_ingestion_complete,
                source_type_weights=retrieval.source_type_weights or {},
            )
            self._structured_ingestion = analyzed_method._service_ref()
            self._analyzed_method = analyzed_method
            if analyzed_method._vision is None:
                logger.warning("no vision provider on AnalyzedIngestion — structured PDF analysis disabled")

        if drawing_method is not None and metadata_store is not None:
            drawing_method.bind(
                metadata_store=metadata_store,
                delegate_methods=list(standard_methods),
            )
            self._drawing_ingestion = drawing_method._service_ref()
            self._drawing_method = drawing_method
            logger.info("drawing ingestion: enabled (via DrawingIngestion method)")

        self._retrieval_service = RetrievalService(
            retrieval_methods=retrieval_methods,
            reranking=retrieval.reranker,
            top_k=retrieval.top_k,
            source_type_weights=retrieval.source_type_weights,
        )

        # Collection-scoped pipelines. Populate both maps symmetrically for every
        # collection — including the first (which reuses the default services) —
        # so retrieval/ingestion against a named collection never silently falls
        # back to the default pipeline.
        self._retrieval_by_collection.clear()
        self._ingestion_by_collection.clear()
        if self._vector_store and hasattr(self._vector_store, "collections"):
            store_collections: list[str] = self._vector_store.collections
            for coll_name in store_collections:
                # INTENTIONAL: the first collection reuses the already-built default service
                # instances. Consequence: retrieving explicitly by the first collection's
                # NAME is equivalent to unscoped retrieval against the default pipeline —
                # they share the same RetrievalService / IngestionService instance (and
                # therefore the same BM25 cache, parent-expansion state, etc.). Later
                # collections get fresh scoped pipelines. Removing this branch requires
                # updating test_default_collection_uses_unscoped_default_services.
                if coll_name == store_collections[0]:
                    self._retrieval_by_collection[coll_name] = self._retrieval_service
                    assert self._ingestion_service is not None
                    self._ingestion_by_collection[coll_name] = self._ingestion_service
                    continue

                scoped_store = self._vector_store.scoped(coll_name)  # type: ignore[attr-defined]
                scoped_retrieval = self._build_retrieval_pipeline(scoped_store, retrieval)
                self._retrieval_by_collection[coll_name] = scoped_retrieval
                self._ingestion_by_collection[coll_name] = self._build_ingestion_service(scoped_store)
                logger.info("pipelines built for collection '%s'", coll_name)

        # Generation
        if gen.provider_client:
            relevance_gate_client = gen.relevance_gate_model if gen.relevance_gate_enabled else None
            self._generation_service = GenerationService(
                provider_client=gen.provider_client,
                system_prompt=gen.system_prompt,
                grounding_enabled=gen.grounding_enabled,
                grounding_threshold=gen.grounding_threshold,
                relevance_gate_enabled=gen.relevance_gate_enabled,
                guiding_enabled=gen.guiding_enabled,
                relevance_gate_client=relevance_gate_client,
                chunk_ordering=gen.chunk_ordering,
            )
            logger.info("generation: enabled")
        else:
            logger.info("generation: disabled (retrieval-only mode)")

        self._knowledge_manager = KnowledgeManager(
            vector_store=self._vector_store,
            metadata_store=metadata_store,
            on_source_removed=self._on_source_removed,
            document_store=self._document_store,
            graph_store=self._graph_store,
        )

        stale = await check_embedding_migration(
            metadata_store=metadata_store,
            embedding_model_name=self._embedding_model_name,
        )
        if stale:
            logger.warning("%d sources are stale and need re-ingestion", stale)

        self._initialized = True

        flows = self._enabled_flows()
        logger.info("knowledge engine ready — %s flows enabled", ", ".join(flows) if flows else "none")

    async def shutdown(self) -> None:
        """Cleanup all store connections in reverse-init order."""
        if not self._stores_opened:
            return  # idempotent: no stores were opened, or shutdown already ran
        self._stores_opened = False  # prevent re-entrant teardown on a second call
        # Reverse-init order: vector → graph → document → metadata
        if self._vector_store:
            try:
                await self._vector_store.shutdown()
            except Exception:
                logger.exception("error shutting down vector store")
        if self._graph_store:
            try:
                await self._graph_store.shutdown()
            except Exception:
                logger.exception("error shutting down graph store")
        if self._document_store:
            try:
                await self._document_store.shutdown()
            except Exception:
                logger.exception("error shutting down document store")
        if self._config.metadata_store:
            try:
                await self._config.metadata_store.shutdown()
            except Exception:
                logger.exception("error shutting down metadata store")
        # Null out service refs so post-shutdown access fails cleanly
        self._ingestion_service = None
        self._structured_ingestion = None
        self._drawing_ingestion = None
        self._analyzed_method = None
        self._drawing_method = None
        self._retrieval_service = None
        self._generation_service = None
        self._knowledge_manager = None
        self._retrieval_namespace = None
        self._ingestion_namespace = None
        self._retrieval_by_collection.clear()
        self._ingestion_by_collection.clear()
        self._initialized = False
        logger.info("knowledge engine shut down")

    async def __aenter__(self) -> KnowledgeEngine:
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
    ) -> Source:
        """Ingest a file. Routes to unstructured or structured based on extension."""
        self._check_initialized()
        _validate_metadata(metadata)
        return await self._with_ingest_telemetry(
            knowledge_id=knowledge_id,
            source_type=source_type,
            run=lambda: self._ingest_impl(
                file_path=Path(file_path),
                knowledge_id=knowledge_id,
                source_type=source_type,
                metadata=metadata,
                page_range=page_range,
                resume_from_chunk=resume_from_chunk,
                on_progress=on_progress,
                collection=collection,
            ),
        )

    async def _ingest_impl(
        self,
        *,
        file_path: Path,
        knowledge_id: str | None,
        source_type: str | None,
        metadata: dict[str, Any] | None,
        page_range: str | None,
        resume_from_chunk: int,
        on_progress: Callable[[int, int], Awaitable[None]] | None,
        collection: str | None,
    ) -> Source:
        ext = file_path.suffix.lower()

        # Drawing route: when a DrawingIngestion method is configured, defer to
        # its ``accepts()``. Drawing is checked before analyzed because a .pdf
        # with source_type='drawing' satisfies both wrappers' accepts() — drawing wins.
        drawing_method = self._drawing_method
        drawing_route = drawing_method is not None and drawing_method.accepts(file_path, source_type)
        if drawing_route:
            if self._drawing_ingestion is None:
                raise ValueError(
                    "Drawing ingestion not configured. Add a DrawingIngestion(...) instance to IngestionConfig.methods."
                )
            if collection is not None:
                raise ValueError("collection routing is not supported with drawing ingestion")

            # Status-based resume
            metadata_store = self._config.metadata_store
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

        analyzed_method = self._analyzed_method
        analyzed_route = (
            analyzed_method is not None
            and self._structured_ingestion is not None
            and analyzed_method.accepts(file_path, source_type)
        )
        if analyzed_route:
            assert self._structured_ingestion is not None
            if collection is not None:
                raise ValueError(
                    f"structured ingestion does not support collection routing "
                    f"(got collection={collection!r}, file type={ext!r})"
                )

            # Status-based resume: find any prior ingest for the same file_hash.
            metadata_store = self._config.metadata_store
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

        return source

    async def ingest_text(
        self,
        content: str,
        knowledge_id: str | None = None,
        source_type: str | None = None,
        metadata: dict[str, Any] | None = None,
        collection: str | None = None,
    ) -> Source:
        """Ingest raw text content into the pipeline."""
        self._check_initialized()
        _validate_ingest_content(content)
        _validate_metadata(metadata)

        async def _run() -> Source:
            ingestion_svc = self._get_ingestion(collection)
            return await ingestion_svc.ingest_text(
                content=content, knowledge_id=knowledge_id, source_type=source_type, metadata=metadata
            )

        return await self._with_ingest_telemetry(
            knowledge_id=knowledge_id,
            source_type=source_type,
            run=_run,
        )

    async def _with_ingest_telemetry(
        self,
        *,
        knowledge_id: str | None,
        source_type: str | None,
        run: Callable[[], Awaitable[Source]],
    ) -> Source:
        """Wrap an ingest call: build IngestTelemetryRow, set contextvars,
        emit lifecycle events, and write the row in `finally`."""
        ingest_id = str(uuid.uuid4())
        row = IngestTelemetryRow(
            source_id="",
            ingest_id=ingest_id,
            knowledge_id=knowledge_id,
            source_type=source_type,
            outcome="success",
        )
        obs = self._observability
        obs_token = _set_obs(obs)
        row_token = _set_row(row)
        start = time.perf_counter()
        await obs.emit(
            "ingest.start",
            "ingest started",
            knowledge_id=knowledge_id,
            ingest_id=ingest_id,
        )
        try:
            source = await run()
            row.duration_ms = int((time.perf_counter() - start) * 1000)
            row.source_id = source.source_id
            row.chunks_count = source.chunk_count
            notes = source.metadata.get("ingestion_notes", []) if source.metadata else []
            row.notes_count = len(notes) if isinstance(notes, list) else 0
            if row.notes_count > 0:
                row.outcome = "partial"
                await obs.emit(
                    "ingest.partial",
                    "ingest partially succeeded",
                    level="warn",
                    knowledge_id=knowledge_id,
                    source_id=source.source_id,
                    ingest_id=ingest_id,
                    context={"duration_ms": row.duration_ms, "notes_count": row.notes_count},
                )
            else:
                row.outcome = "success"
                await obs.emit(
                    "ingest.success",
                    "ingest succeeded",
                    knowledge_id=knowledge_id,
                    source_id=source.source_id,
                    ingest_id=ingest_id,
                    context={"duration_ms": row.duration_ms},
                )
            return source
        except BaseException as exc:
            row.duration_ms = int((time.perf_counter() - start) * 1000)
            row.outcome = "error"
            row.error_type = type(exc).__name__
            row.error_message = str(exc)
            await obs.emit(
                "ingest.error",
                "ingest failed",
                level="error",
                knowledge_id=knowledge_id,
                ingest_id=ingest_id,
                context={"duration_ms": row.duration_ms},
                error=exc,
            )
            raise
        finally:
            try:
                await self._telemetry.write(row)
            except Exception:  # noqa: BLE001
                logger.exception("telemetry write failed for ingest_id=%s", ingest_id)
            _reset_row(row_token)
            _reset_obs(obs_token)

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
                "DrawingIngestion not configured — add a DrawingIngestion(...) to IngestionConfig.methods."
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
                "DrawingIngestion not configured — add a DrawingIngestion(...) to IngestionConfig.methods."
            )
        return await self._drawing_ingestion.extract(source_id)

    async def link_drawing(self, source_id: str) -> Source:
        """Drawing phase 3: cross-sheet linking (deterministic + LLM residue)."""
        self._check_initialized()
        if self._drawing_ingestion is None:
            raise ConfigurationError(
                "DrawingIngestion not configured — add a DrawingIngestion(...) to IngestionConfig.methods."
            )
        return await self._drawing_ingestion.link(source_id)

    async def complete_drawing_ingestion(self, source_id: str) -> Source:
        """Drawing phase 4: embed + graph write."""
        self._check_initialized()
        if self._drawing_ingestion is None:
            raise ConfigurationError(
                "DrawingIngestion not configured — add a DrawingIngestion(...) to IngestionConfig.methods."
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

        RETRIEVAL (default) runs retrieve-then-generate. DIRECT loads the
        entire corpus into the prompt and skips retrieval. AUTO picks
        DIRECT or RETRIEVAL per query based on corpus size
        (`RoutingConfig.full_context_threshold`).
        """
        self._check_initialized()
        _validate_query_text(text)
        if not self._generation_service:
            raise ConfigurationError("query() requires generation.provider_client to be configured")

        mode = self._config.routing.mode
        row = QueryTelemetryRow(
            query_id=str(uuid.uuid4()),
            knowledge_id=knowledge_id,
            mode="retrieval" if mode == QueryMode.RETRIEVAL else "direct",
            routing_decision=mode.name.lower(),
            outcome="success",
        )
        obs = self._observability
        obs_token = _set_obs(obs)
        row_token = _set_row(row)
        start = time.perf_counter()
        await obs.emit(
            "query.start",
            "query started",
            knowledge_id=knowledge_id,
            query_id=row.query_id,
        )
        try:
            if mode == QueryMode.RETRIEVAL:
                await obs.emit(
                    "routing.decision",
                    "explicit retrieval mode",
                    context={
                        "mode": "retrieval",
                        "corpus_tokens": None,
                        "threshold": None,
                        "reason": "explicit_mode",
                    },
                )
                result = await self._query_via_retrieval(
                    text, knowledge_id, history, min_score, collection, system_prompt, trace
                )
            elif mode == QueryMode.DIRECT:
                await obs.emit(
                    "routing.decision",
                    "explicit direct mode",
                    context={
                        "mode": "direct",
                        "corpus_tokens": None,
                        "threshold": None,
                        "reason": "explicit_mode",
                    },
                )
                result = await self._query_via_direct_context(text, knowledge_id, history, system_prompt, trace)
            else:
                result = await self._query_via_auto(
                    text, knowledge_id, history, min_score, collection, system_prompt, trace, row=row
                )
            row.duration_ms = int((time.perf_counter() - start) * 1000)
            row.confidence = result.confidence
            if result.clarification is not None:
                row.grounding_decision = "clarification"
                row.outcome = "refused"
                await obs.emit(
                    "query.refused",
                    "query refused (clarification)",
                    knowledge_id=knowledge_id,
                    query_id=row.query_id,
                    context={"duration_ms": row.duration_ms},
                )
            elif result.grounded:
                row.grounding_decision = "grounded"
                row.outcome = "success"
                await obs.emit(
                    "query.success",
                    "query succeeded",
                    knowledge_id=knowledge_id,
                    query_id=row.query_id,
                    context={"duration_ms": row.duration_ms},
                )
            else:
                row.grounding_decision = "ungrounded"
                row.outcome = "refused"
                await obs.emit(
                    "query.refused",
                    "query refused (ungrounded)",
                    knowledge_id=knowledge_id,
                    query_id=row.query_id,
                    context={"duration_ms": row.duration_ms},
                )
            return result
        except BaseException as exc:
            row.duration_ms = int((time.perf_counter() - start) * 1000)
            row.outcome = "error"
            row.error_type = type(exc).__name__
            row.error_message = str(exc)
            await obs.emit(
                "query.error",
                "query failed",
                level="error",
                knowledge_id=knowledge_id,
                query_id=row.query_id,
                context={"duration_ms": row.duration_ms},
                error=exc,
            )
            raise
        finally:
            try:
                await self._telemetry.write(row)
            except Exception:  # noqa: BLE001
                logger.exception("telemetry write failed for query_id=%s", row.query_id)
            _reset_row(row_token)
            _reset_obs(obs_token)

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
        """Retrieve-then-generate pipeline (RETRIEVAL mode)."""
        assert self._generation_service is not None
        chunks, trace_obj = await self._retrieve_chunks(text, knowledge_id, history, min_score, collection, trace=trace)

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

    async def _query_via_auto(
        self,
        text: str,
        knowledge_id: str | None,
        history: list[tuple[str, str]] | None,
        min_score: float | None,
        collection: str | None,
        system_prompt: str | None,
        trace: bool,
        *,
        row: QueryTelemetryRow | None = None,
    ) -> QueryResult:
        """AUTO: pick DIRECT or RETRIEVAL per query based on corpus token count."""
        assert self._knowledge_manager is not None
        tokens = await self._knowledge_manager.get_corpus_tokens(knowledge_id)
        threshold = self._config.routing.full_context_threshold
        decision = "direct" if tokens <= threshold else "retrieval"

        if row is not None:
            row.routing_decision = decision
            row.mode = decision  # type: ignore[assignment]
            row.corpus_tokens = tokens

        await self._observability.emit(
            "routing.decision",
            f"auto routing: corpus_tokens={tokens} threshold={threshold} -> {decision}",
            context={
                "mode": decision,
                "corpus_tokens": tokens,
                "threshold": threshold,
                "reason": "auto_dispatch",
            },
        )

        if decision == "direct":
            logger.info("auto routing: tokens=%d threshold=%d → DIRECT", tokens, threshold)
            return await self._query_via_direct_context(text, knowledge_id, history, system_prompt, trace)

        logger.info("auto routing: tokens=%d threshold=%d → RETRIEVAL", tokens, threshold)
        return await self._query_via_retrieval(text, knowledge_id, history, min_score, collection, system_prompt, trace)

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
            raise ConfigurationError("query_stream() requires generation.provider_client to be configured")

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
        element is ``None`` so callers may unpack ``chunks, _ = await engine.retrieve(...)``
        for chunks-only access. With ``trace=True``, returns the
        :class:`RetrievalTrace` with ``grounding_decision`` and ``confidence``
        left as ``None`` (no generation/grounding stage runs in raw retrieval);
        ``final_results`` carries the post-min-score-filter chunks.
        """
        self._check_initialized()
        _validate_query_text(text)
        chunks, trace_obj = await self._retrieve_chunks(text, knowledge_id, None, min_score, collection, trace=trace)
        return chunks, trace_obj

    async def benchmark(
        self,
        cases: list[BenchmarkCase],
        config: BenchmarkConfig | None = None,
        knowledge_id: str | None = None,
        llm_judge: LLMJudgment | None = None,
    ) -> BenchmarkReport:
        """Run cases, collect traces and failure classifications, aggregate.

        Each case is executed via `query(..., trace=True)` so the report
        carries the full per-case trace alongside the aggregate metrics.
        Concurrency is bounded by `config.concurrency` (default 1, serial).
        Pass `llm_judge` only when paying per-case judge LLM cost is
        intended; the judge is `None` in the report when omitted.
        """
        self._check_initialized()
        if not self._generation_service:
            raise ConfigurationError("benchmark() requires generation.provider_client to be configured")

        async def _run_one(query_text: str, *, trace: bool) -> QueryResult:
            return await self.query(query_text, knowledge_id=knowledge_id, trace=trace)

        return await run_benchmark(cases, _run_one, config=config, llm_judge=llm_judge)

    def _embeddings_from_methods(self) -> BaseEmbeddings | None:
        """Walk the configured ingestion methods and return the first
        embeddings instance found (SemanticIngestion / AnalyzedIngestion /
        DrawingIngestion all expose ``_embeddings``)."""
        return next(
            (m._embeddings for m in self._config.ingestion.methods if hasattr(m, "_embeddings")),
            None,
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts using the configured provider."""
        self._check_initialized()
        embeddings = self._embeddings_from_methods()
        if embeddings is None:
            raise ConfigurationError("embed() requires an ingestion method that carries embeddings")
        for text in texts:
            _validate_query_text(text)
        return await embed_batched(embeddings, texts)

    async def embed_single(self, text: str) -> list[float]:
        """Generate an embedding for a single text."""
        self._check_initialized()
        embeddings = self._embeddings_from_methods()
        if embeddings is None:
            raise ConfigurationError("embed_single() requires an ingestion method that carries embeddings")
        _validate_query_text(text)
        vectors = await embed_batched(embeddings, [text])
        return vectors[0]

    async def _load_full_corpus(self, knowledge_id: str | None) -> str:
        """Concatenate every source's text under ``knowledge_id`` into one string.

        Plumbing for FULL_CONTEXT mode — it needs the whole corpus in-prompt,
        not retrieval-ranked chunks. The model-context-limit check is enforced
        upfront at engine init by
        ``_validate_full_context_fits_provider_window`` when the generation
        provider declares ``context_size``; this method does not truncate.

        Strategy per source: prefer the document store (lossless, original
        text). Fall back to a vector-store scroll only when the document store
        is absent or returns nothing — the vector path is lossy because chunk
        boundaries can land mid-sentence and table-row chunks are flattened
        into linear text. Skips parent chunks (`chunk_type == "parent"`) on
        the scroll path so parent-child indexing doesn't double the emitted
        text.

        Sequential per-source: a knowledge with N sources issues N reads.
        Acceptable in practice — DIRECT mode invokes this once per query, not
        per chunk. Batch fetch via `IN (?, ?, ...)` is straightforward to add
        if a real workload exposes the latency.

        Returns the empty string when the knowledge has no sources.
        """
        metadata_store = self._config.metadata_store
        if metadata_store is None:
            return ""

        sources = await metadata_store.list_sources(knowledge_id=knowledge_id)
        # Pin the corpus prefix to a stable function of WHICH sources are
        # present, not WHEN they were ingested. The metadata store today orders
        # by `created_at DESC`, which means re-ingesting a source bumps it to
        # the front and busts the entire prompt cache. Sorting by source_id
        # here makes re-ingestion cache-invalidate only the affected slot, and
        # adding a new source only changes the suffix below existing ones.
        sources = sorted(sources, key=lambda s: s.source_id)
        document_store = self._document_store
        vector_store = self._vector_store
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
        parent-child indexing doesn't double-count text.
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

    async def _invalidate_keyword_caches(self, knowledge_id: str | None) -> None:
        """Invalidate BM25 cache on every scoped collection's SemanticRetrieval, not
        just the default one. Scoped collections carry their own method instances
        with their own caches — iterate them all so stale BM25 results are not
        returned after a cross-collection ingest/remove."""
        seen: set[int] = set()
        if self._retrieval_by_collection:
            # SERIAL: invalidate_cache mutates in-memory BM25 state; the `seen`
            # set prevents double-invalidation of shared method instances across
            # collections. Concurrent invalidations on the same method instance
            # would require locking the BM25 cache — serial is simpler and safe.
            for retrieval_service in self._retrieval_by_collection.values():
                for method in retrieval_service.methods:
                    if method.name != "keyword" or id(method) in seen:
                        continue
                    seen.add(id(method))
                    if hasattr(method, "invalidate_cache"):
                        await method.invalidate_cache(knowledge_id)
            return
        # No multi-collection wiring — fall back to the default namespace.
        if self._retrieval_namespace and "keyword" in self._retrieval_namespace:
            keyword = self._retrieval_namespace.keyword
            if hasattr(keyword, "invalidate_cache"):
                await keyword.invalidate_cache(knowledge_id)

    async def _on_ingestion_complete(self, knowledge_id: str | None) -> None:
        """Callback after ingestion — invalidates the keyword (BM25) cache for the knowledge_id."""
        await self._invalidate_keyword_caches(knowledge_id)

    async def _on_source_removed(self, knowledge_id: str | None) -> None:
        """Callback after source removal — invalidates BM25 cache for the knowledge_id."""
        await self._invalidate_keyword_caches(knowledge_id)

    def _build_retrieval_pipeline(
        self,
        vector_store: BaseVectorStore,
        retrieval: RetrievalConfig,
    ) -> RetrievalService:
        methods: list = []
        for m in retrieval.methods:
            if isinstance(m, SemanticRetrieval) or isinstance(m, KeywordRetrieval) and m.backend == "bm25":
                methods.append(m.clone_for_store(vector_store))
            else:
                methods.append(m)

        return RetrievalService(
            retrieval_methods=methods,
            reranking=retrieval.reranker,
            top_k=retrieval.top_k,
            source_type_weights=retrieval.source_type_weights,
        )

    def _build_ingestion_service(self, vector_store: BaseVectorStore) -> IngestionService:
        assert self._chunker is not None
        cfg = self._config
        methods: list = []
        vision_parser: BaseVision | None = None
        for m in cfg.ingestion.methods:
            if isinstance(m, SemanticIngestion):
                methods.append(m.clone_for_store(vector_store))
            else:
                methods.append(m)
            if vision_parser is None and isinstance(m, AnalyzedIngestion | DrawingIngestion):
                vision_parser = m._vision

        return IngestionService(
            chunker=self._chunker,
            ingestion_methods=methods,
            embedding_model_name=self._embedding_model_name,
            source_type_weights=cfg.retrieval.source_type_weights,
            metadata_store=cfg.metadata_store,
            on_ingestion_complete=self._on_ingestion_complete,
            vision_parser=vision_parser,
            chunk_context_headers=cfg.ingestion.chunk_context_headers,
            document_expansion=cfg.ingestion.document_expansion,
            expansion_registry=self._expansion_registry,
            contextual_chunk=cfg.ingestion.contextual_chunk,
            token_counter=cfg.ingestion.token_counter,
        )

    async def _retrieve_chunks(
        self,
        text: str,
        knowledge_id: str | None,
        history: list[tuple[str, str]] | None,
        min_score: float | None,
        collection: str | None,
        trace: bool = False,
        top_k: int | None = None,
    ) -> tuple[list[RetrievedChunk], RetrievalTrace | None]:
        """Shared retrieval over the three pillars + score filter.

        When ``trace=True``, returns the trace alongside the chunks. The
        trace's ``final_results`` reflect the post-min-score-filter view.

        ``top_k`` overrides ``RetrievalConfig.top_k`` for this call only.
        """
        retrieval_service = self._get_retrieval(collection)
        retrieval_query = self._build_retrieval_query(text, history)

        extra_kwargs: dict[str, Any] = {}
        if top_k is not None:
            extra_kwargs["top_k"] = top_k

        chunks, trace_obj = await retrieval_service.retrieve(
            query=retrieval_query, knowledge_id=knowledge_id, trace=trace, **extra_kwargs
        )

        if min_score is not None:
            chunks = [c for c in chunks if c.score >= min_score]

        if trace_obj is not None:
            trace_obj.final_results = list(chunks)

        return chunks, trace_obj

    def _get_retrieval(self, collection: str | None) -> RetrievalService:
        """Return retrieval pipeline for *collection* (default if None).

        Raises ``ValueError`` when *collection* is specified but unknown —
        previously this silently fell back to the default pipeline, which
        could mix data across collections without any warning.
        """
        if collection is None:
            assert self._retrieval_service is not None
            return self._retrieval_service
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
                "KnowledgeEngine not initialized. Use 'async with KnowledgeEngine(config) as engine:' "
                "or call 'await engine.initialize()' first."
            )

    def _enabled_flows(self) -> list[str]:
        flows: list[str] = []
        if self._retrieval_namespace:
            flows.extend(m.name for m in self._retrieval_namespace)
        if self._structured_ingestion:
            flows.append("structured")
        if self._generation_service:
            flows.append("generation")
        return flows
