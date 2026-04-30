from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any

from rfnry_rag.config.drawing import DrawingIngestionConfig as DrawingIngestionConfig
from rfnry_rag.config.engine import RagEngineConfig
from rfnry_rag.config.generation import DEFAULT_SYSTEM_PROMPT as DEFAULT_SYSTEM_PROMPT
from rfnry_rag.config.generation import GenerationConfig as GenerationConfig
from rfnry_rag.config.graph import GraphIngestionConfig as GraphIngestionConfig
from rfnry_rag.config.ingestion import DocumentExpansionConfig as DocumentExpansionConfig
from rfnry_rag.config.ingestion import IngestionConfig
from rfnry_rag.config.retrieval import RetrievalConfig
from rfnry_rag.config.routing import QueryMode
from rfnry_rag.config.routing import RoutingConfig as RoutingConfig
from rfnry_rag.exceptions import ConfigurationError, InputError
from rfnry_rag.generation.models import QueryResult, StepResult, StreamEvent
from rfnry_rag.generation.service import GenerationService
from rfnry_rag.generation.step import StepGenerationService
from rfnry_rag.ingestion.analyze.service import AnalyzedIngestionService
from rfnry_rag.ingestion.base import BaseIngestionMethod
from rfnry_rag.ingestion.chunk.chunker import SemanticChunker
from rfnry_rag.ingestion.chunk.service import IngestionService
from rfnry_rag.ingestion.drawing.service import DrawingIngestionService
from rfnry_rag.ingestion.embeddings.base import BaseEmbeddings
from rfnry_rag.ingestion.embeddings.sparse.base import BaseSparseEmbeddings
from rfnry_rag.ingestion.hashing import file_hash as compute_file_hash
from rfnry_rag.ingestion.methods.document import DocumentIngestion
from rfnry_rag.ingestion.methods.graph import GraphIngestion
from rfnry_rag.ingestion.methods.vector import VectorIngestion
from rfnry_rag.knowledge.manager import KnowledgeManager
from rfnry_rag.knowledge.migration import check_embedding_migration
from rfnry_rag.logging import get_logger
from rfnry_rag.models import RetrievedChunk, Source
from rfnry_rag.observability.benchmark import (
    BenchmarkCase,
    BenchmarkConfig,
    BenchmarkReport,
    run_benchmark,
)
from rfnry_rag.observability.metrics import LLMJudgment
from rfnry_rag.observability.trace import RetrievalTrace
from rfnry_rag.providers import build_registry
from rfnry_rag.retrieval.base import BaseRetrievalMethod
from rfnry_rag.retrieval.enrich.service import StructuredRetrievalService
from rfnry_rag.retrieval.methods.document import DocumentRetrieval
from rfnry_rag.retrieval.methods.graph import GraphRetrieval
from rfnry_rag.retrieval.methods.vector import VectorRetrieval
from rfnry_rag.retrieval.namespace import MethodNamespace
from rfnry_rag.retrieval.search.reranking.base import BaseReranking
from rfnry_rag.retrieval.search.rewriting.base import BaseQueryRewriting
from rfnry_rag.retrieval.search.service import RetrievalService
from rfnry_rag.stores.document.base import BaseDocumentStore
from rfnry_rag.stores.graph.base import BaseGraphStore
from rfnry_rag.stores.vector.base import BaseVectorStore

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
        embedding_model_name: str = "",
        metadata_store: Any = None,
        top_k: int = 5,
        reranker: BaseReranking | None = None,
        query_rewriter: BaseQueryRewriting | None = None,
        sparse_embeddings: BaseSparseEmbeddings | None = None,
    ) -> RagEngineConfig:
        """Preset: dense vector search only."""
        name = embedding_model_name or _derive_embedding_model_name(embeddings)
        return RagEngineConfig(
            metadata_store=metadata_store,
            ingestion=IngestionConfig(
                methods=[
                    VectorIngestion(
                        store=vector_store,
                        embeddings=embeddings,
                        embedding_model_name=name,
                        sparse_embeddings=sparse_embeddings,
                    )
                ],
                embeddings=embeddings,
                sparse_embeddings=sparse_embeddings,
            ),
            retrieval=RetrievalConfig(
                methods=[
                    VectorRetrieval(store=vector_store, embeddings=embeddings, sparse_embeddings=sparse_embeddings)
                ],
                top_k=top_k,
                reranker=reranker,
                query_rewriter=query_rewriter,
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
    ) -> RagEngineConfig:
        """Preset: full-text / substring search only. No embeddings needed."""
        return RagEngineConfig(
            metadata_store=metadata_store,
            ingestion=IngestionConfig(methods=[DocumentIngestion(store=document_store)]),
            retrieval=RetrievalConfig(
                methods=[DocumentRetrieval(store=document_store, weight=0.8)],
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
        query_rewriter: BaseQueryRewriting | None = None,
        top_k: int = 5,
    ) -> RagEngineConfig:
        """Preset: multi-path retrieval with optional document/graph/sparse paths + rerank."""
        name = _derive_embedding_model_name(embeddings)
        ing_methods: list[Any] = [
            VectorIngestion(
                store=vector_store,
                embeddings=embeddings,
                embedding_model_name=name,
                sparse_embeddings=sparse_embeddings,
            )
        ]
        ret_methods: list[Any] = [
            VectorRetrieval(store=vector_store, embeddings=embeddings, sparse_embeddings=sparse_embeddings)
        ]
        if document_store is not None:
            ing_methods.append(DocumentIngestion(store=document_store))
            ret_methods.append(DocumentRetrieval(store=document_store, weight=0.8))
        if graph_store is not None:
            ret_methods.append(GraphRetrieval(store=graph_store, weight=0.7))

        return RagEngineConfig(
            metadata_store=metadata_store,
            ingestion=IngestionConfig(methods=ing_methods, embeddings=embeddings, sparse_embeddings=sparse_embeddings),
            retrieval=RetrievalConfig(
                methods=ret_methods, top_k=top_k, reranker=reranker, query_rewriter=query_rewriter
            ),
        )

    def __init__(self, config: RagEngineConfig) -> None:
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

        self._retrieval_namespace: MethodNamespace[BaseRetrievalMethod] | None = None
        self._ingestion_namespace: MethodNamespace[BaseIngestionMethod] | None = None

        self._retrieval_by_collection: dict[str, tuple[RetrievalService, StructuredRetrievalService | None]] = {}
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
            if isinstance(m, VectorIngestion | VectorRetrieval) and self._vector_store is None:
                self._vector_store = store
            elif isinstance(m, DocumentIngestion | DocumentRetrieval) and self._document_store is None:
                self._document_store = store
            elif isinstance(m, GraphIngestion | GraphRetrieval) and self._graph_store is None:
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
        """Cross-config validation: ensure at least one retrieval method is configured."""
        cfg = self._config
        if not cfg.retrieval.methods:
            raise ConfigurationError(
                "RetrievalConfig.methods must not be empty — configure at least one "
                "retrieval method (VectorRetrieval, DocumentRetrieval, or GraphRetrieval)."
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
        ingestion = cfg.ingestion
        retrieval = cfg.retrieval
        gen = cfg.generation
        metadata_store = cfg.metadata_store

        logger.info("ragengine initializing")

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

        # Vector store init: ask the first VectorIngestion/VectorRetrieval method
        # for its embedding dimension and pre-create the store collection.
        if self._vector_store is not None and ingestion.embeddings is not None:
            vector_size = await ingestion.embeddings.embedding_dimension()
            await self._vector_store.initialize(vector_size)
            self._embedding_model_name = _derive_embedding_model_name(ingestion.embeddings)

        # Chunker
        self._chunker = SemanticChunker(
            chunk_size=ingestion.chunk_size,
            chunk_overlap=ingestion.chunk_overlap,
            parent_chunk_size=ingestion.parent_chunk_size,
            parent_chunk_overlap=ingestion.parent_chunk_overlap,
            chunk_size_unit=ingestion.chunk_size_unit,
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
            build_registry(ingestion.document_expansion.lm_client)
            if ingestion.document_expansion.enabled and ingestion.document_expansion.lm_client
            else None
        )

        # Build services
        self._ingestion_service = IngestionService(
            chunker=self._chunker,
            ingestion_methods=ingestion_methods,
            embedding_model_name=self._embedding_model_name,
            source_type_weights=retrieval.source_type_weights,
            metadata_store=metadata_store,
            on_ingestion_complete=self._on_ingestion_complete,
            vision_parser=ingestion.vision,
            chunk_context_headers=ingestion.chunk_context_headers,
            document_expansion=ingestion.document_expansion,
            expansion_registry=self._expansion_registry,
        )

        # Analyzed ingestion — shares document method from main list, graph store passed directly
        if metadata_store and self._vector_store and ingestion.embeddings:
            analyzed_methods = [m for m in ingestion_methods if isinstance(m, DocumentIngestion)]
            if not analyzed_methods:
                logger.warning(
                    "structured ingestion enabled but no DocumentIngestion configured — "
                    "analyzed phase 3 will skip document storage"
                )

            self._structured_ingestion = AnalyzedIngestionService(
                embeddings=ingestion.embeddings,
                vector_store=self._vector_store,
                metadata_store=metadata_store,
                embedding_model_name=self._embedding_model_name,
                vision=ingestion.vision,
                dpi=ingestion.dpi,
                source_type_weights=retrieval.source_type_weights,
                on_ingestion_complete=self._on_ingestion_complete,
                lm_client=ingestion.lm_client,
                graph_store=self._graph_store,
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
            if metadata_store is None or self._vector_store is None:
                raise ConfigurationError(
                    "DrawingIngestionConfig.enabled=True requires metadata_store and a vector store"
                )
            if ingestion.embeddings is None:
                raise ConfigurationError("DrawingIngestionConfig.enabled=True requires IngestionConfig.embeddings")
            self._drawing_ingestion = DrawingIngestionService(
                config=ingestion.drawings,
                embeddings=ingestion.embeddings,
                vector_store=self._vector_store,
                metadata_store=metadata_store,
                embedding_model_name=self._embedding_model_name,
                graph_store=self._graph_store,
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
        if self._vector_store and ingestion.embeddings:
            self._structured_retrieval = StructuredRetrievalService(
                vector_store=self._vector_store,
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
                    self._retrieval_by_collection[coll_name] = (
                        self._retrieval_service,
                        self._structured_retrieval,
                    )
                    assert self._ingestion_service is not None
                    self._ingestion_by_collection[coll_name] = self._ingestion_service
                    continue

                scoped_store = self._vector_store.scoped(coll_name)  # type: ignore[attr-defined]
                scoped_retrieval = self._build_retrieval_pipeline(scoped_store, retrieval)
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
        logger.info("ragengine ready — %s flows enabled", ", ".join(flows) if flows else "none")

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
        self._retrieval_service = None
        self._structured_retrieval = None
        self._generation_service = None
        self._knowledge_manager = None
        self._step_service = None
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
    ) -> Source:
        """Ingest a file. Routes to unstructured or structured based on extension."""
        self._check_initialized()
        _validate_metadata(metadata)
        file_path = Path(file_path)
        ext = file_path.suffix.lower()

        # Drawing route: .dxf always; .pdf only when source_type='drawing'.
        drawing_route = ext in SUPPORTED_DRAWING_EXTENSIONS or (ext == ".pdf" and source_type == "drawing")
        if drawing_route:
            if self._drawing_ingestion is None:
                raise ValueError(
                    "Drawing ingestion not configured. "
                    "Pass IngestionConfig(drawings=DrawingIngestionConfig(enabled=True, ...))."
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

        if ext in SUPPORTED_STRUCTURED_EXTENSIONS and self._structured_ingestion:
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
            raise ConfigurationError("DrawingIngestionConfig not configured — pass IngestionConfig(drawings=...).")
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
            raise ConfigurationError("DrawingIngestionConfig not configured — pass IngestionConfig(drawings=...).")
        return await self._drawing_ingestion.extract(source_id)

    async def link_drawing(self, source_id: str) -> Source:
        """Drawing phase 3: cross-sheet linking (deterministic + LLM residue)."""
        self._check_initialized()
        if self._drawing_ingestion is None:
            raise ConfigurationError("DrawingIngestionConfig not configured — pass IngestionConfig(drawings=...).")
        return await self._drawing_ingestion.link(source_id)

    async def complete_drawing_ingestion(self, source_id: str) -> Source:
        """Drawing phase 4: embed + graph write."""
        self._check_initialized()
        if self._drawing_ingestion is None:
            raise ConfigurationError("DrawingIngestionConfig not configured — pass IngestionConfig(drawings=...).")
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
            raise ConfigurationError("query() requires generation.lm_client to be configured")

        mode = self._config.routing.mode
        if mode == QueryMode.INDEXED:
            return await self._query_via_retrieval(
                text, knowledge_id, history, min_score, collection, system_prompt, trace
            )
        if mode == QueryMode.FULL_CONTEXT:
            return await self._query_via_direct_context(text, knowledge_id, history, system_prompt, trace)
        return await self._query_via_auto(text, knowledge_id, history, min_score, collection, system_prompt, trace)

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
            trace_obj.routing_decision = "indexed"
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
                routing_decision="full_context",
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
    ) -> QueryResult:
        """AUTO: pick DIRECT or RETRIEVAL per query based on corpus token count."""
        assert self._knowledge_manager is not None
        tokens = await self._knowledge_manager.get_corpus_tokens(knowledge_id)
        threshold = self._config.routing.full_context_threshold

        if tokens <= threshold:
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
            raise ConfigurationError("query_stream() requires generation.lm_client to be configured")

        mode = self._config.routing.mode
        if mode != QueryMode.INDEXED:
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
        chunks, trace_obj = await self._retrieve_chunks(text, knowledge_id, None, min_score, collection, trace=trace)
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
        """Run cases, collect traces and failure classifications, aggregate.

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

        Plumbing for DIRECT / HYBRID modes — they need the whole corpus
        in-prompt, not retrieval-ranked chunks. Caller is responsible for the
        downstream model-context-limit check; this method does not truncate.

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
        retrieval: RetrievalConfig,
    ) -> tuple[RetrievalService, StructuredRetrievalService | None]:
        methods: list = []
        embeddings: BaseEmbeddings | None = None
        for m in retrieval.methods:
            if isinstance(m, VectorRetrieval):
                methods.append(m.clone_for_store(vector_store))
                embeddings = m._embeddings
            else:
                methods.append(m)

        unstructured = RetrievalService(
            retrieval_methods=methods,
            reranking=retrieval.reranker,
            top_k=retrieval.top_k,
            source_type_weights=retrieval.source_type_weights,
            query_rewriter=retrieval.query_rewriter,
            chunk_refiner=retrieval.chunk_refiner,
        )
        structured: StructuredRetrievalService | None = None
        if embeddings is not None:
            structured = StructuredRetrievalService(
                vector_store=vector_store,
                embeddings=embeddings,
                lm_client=retrieval.enrich_lm_client,
                top_k=retrieval.top_k,
                enrich_cross_references=retrieval.cross_reference_enrichment,
            )
        return unstructured, structured

    def _build_ingestion_service(self, vector_store: BaseVectorStore) -> IngestionService:
        assert self._chunker is not None
        cfg = self._config
        methods: list = []
        for m in cfg.ingestion.methods:
            if isinstance(m, VectorIngestion):
                methods.append(m.clone_for_store(vector_store))
            else:
                methods.append(m)

        return IngestionService(
            chunker=self._chunker,
            ingestion_methods=methods,
            embedding_model_name=self._embedding_model_name,
            source_type_weights=cfg.retrieval.source_type_weights,
            metadata_store=cfg.metadata_store,
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
        top_k: int | None = None,
    ) -> tuple[list[RetrievedChunk], RetrievalTrace | None]:
        """Shared retrieval: unstructured + structured merge + score filter.

        When `trace=True`, returns the unstructured-pipeline trace alongside
        the merged chunks. The trace's `final_results` reflect the
        post-min-score-filter view returned to the caller.

        `top_k` overrides the configured `RetrievalConfig.top_k` for this
        call only; passed through to `RetrievalService.retrieve`. The
        confidence-expansion loop uses this to retry with `top_k * 2`
        without mutating the service-level default.
        """
        unstructured, structured = self._get_retrieval(collection)
        retrieval_query = self._build_retrieval_query(text, history)

        extra_kwargs: dict[str, Any] = {}
        if top_k is not None:
            extra_kwargs["top_k"] = top_k

        trace_obj: RetrievalTrace | None = None
        if structured:
            results = await asyncio.gather(
                unstructured.retrieve(query=retrieval_query, knowledge_id=knowledge_id, trace=trace, **extra_kwargs),
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
                query=retrieval_query, knowledge_id=knowledge_id, trace=trace, **extra_kwargs
            )

        if min_score is not None:
            chunks = [c for c in chunks if c.score >= min_score]

        if trace_obj is not None:
            trace_obj.final_results = list(chunks)

        return chunks, trace_obj

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
        from rfnry_rag.retrieval.search.fusion import reciprocal_rank_fusion

        return reciprocal_rank_fusion([unstructured, structured])
