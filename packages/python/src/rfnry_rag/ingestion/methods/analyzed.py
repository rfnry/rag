from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from rfnry_rag.config.graph import GraphIngestionConfig
from rfnry_rag.exceptions import ConfigurationError
from rfnry_rag.ingestion.analyze.service import AnalyzedIngestionService
from rfnry_rag.ingestion.embeddings.base import BaseEmbeddings
from rfnry_rag.ingestion.vision.base import BaseVision
from rfnry_rag.models import Source
from rfnry_rag.providers import GenerativeModelClient
from rfnry_rag.stores.graph.base import BaseGraphStore
from rfnry_rag.stores.metadata.base import BaseMetadataStore
from rfnry_rag.stores.vector.base import BaseVectorStore


class AnalyzedIngestion:
    """Method wrapper around AnalyzedIngestionService."""

    required: bool = True

    def __init__(
        self,
        store: BaseVectorStore,
        embeddings: BaseEmbeddings,
        vision: BaseVision | None = None,
        lm_client: GenerativeModelClient | None = None,
        graph_store: BaseGraphStore | None = None,
        metadata_store: BaseMetadataStore | None = None,
        embedding_model_name: str = "",
        dpi: int = 300,
        analyze_text_skip_threshold_chars: int = 300,
        analyze_concurrency: int = 5,
        graph_config: GraphIngestionConfig | None = None,
        source_type_weights: dict[str, float] | None = None,
        on_ingestion_complete: Callable[[str | None], Awaitable[None]] | None = None,
        delegate_methods: list[Any] | None = None,
    ) -> None:
        if not (72 <= dpi <= 600):
            raise ConfigurationError(f"AnalyzedIngestion.dpi={dpi} out of range [72, 600]")
        if not (1 <= analyze_concurrency <= 100):
            raise ConfigurationError(
                f"AnalyzedIngestion.analyze_concurrency={analyze_concurrency} out of range [1, 100]"
            )
        if not (0 <= analyze_text_skip_threshold_chars <= 100_000):
            raise ConfigurationError(
                f"AnalyzedIngestion.analyze_text_skip_threshold_chars={analyze_text_skip_threshold_chars} "
                "out of range [0, 100_000]"
            )
        self._store = store
        if not embedding_model_name:
            embedding_model_name = getattr(embeddings, "name", "") or ""
        self._embeddings = embeddings
        self._vision = vision
        self._lm_client = lm_client
        self._graph_store = graph_store
        self._metadata_store = metadata_store
        self._embedding_model_name = embedding_model_name
        self._dpi = dpi
        self._analyze_text_skip_threshold_chars = analyze_text_skip_threshold_chars
        self._analyze_concurrency = analyze_concurrency
        self._graph_config = graph_config
        self._source_type_weights = source_type_weights or {}
        self._on_ingestion_complete = on_ingestion_complete
        self._delegate_methods = list(delegate_methods or [])
        self._service: AnalyzedIngestionService | None = None

    @property
    def name(self) -> str:
        return "analyzed"

    def accepts(self, file_path: Path, source_type: str | None) -> bool:
        return file_path.suffix.lower() in {".pdf", ".xml", ".l5x"}

    def clone_for_store(self, store: BaseVectorStore) -> AnalyzedIngestion:
        return AnalyzedIngestion(
            store=store,
            embeddings=self._embeddings,
            vision=self._vision,
            lm_client=self._lm_client,
            graph_store=self._graph_store,
            metadata_store=self._metadata_store,
            embedding_model_name=self._embedding_model_name,
            dpi=self._dpi,
            analyze_text_skip_threshold_chars=self._analyze_text_skip_threshold_chars,
            analyze_concurrency=self._analyze_concurrency,
            graph_config=self._graph_config,
            source_type_weights=self._source_type_weights,
            on_ingestion_complete=self._on_ingestion_complete,
            delegate_methods=self._delegate_methods,
        )

    def bind(
        self,
        metadata_store: BaseMetadataStore,
        delegate_methods: list[Any],
        on_ingestion_complete: Callable[[str | None], Awaitable[None]] | None,
        source_type_weights: dict[str, float],
    ) -> None:
        self._metadata_store = metadata_store
        self._delegate_methods = list(delegate_methods)
        self._on_ingestion_complete = on_ingestion_complete
        self._source_type_weights = dict(source_type_weights)
        self._service = None

    def _build_service(self) -> AnalyzedIngestionService:
        if self._metadata_store is None:
            raise RuntimeError("AnalyzedIngestion not bound — call bind() first")
        return AnalyzedIngestionService(
            embeddings=self._embeddings,
            vector_store=self._store,
            metadata_store=self._metadata_store,
            embedding_model_name=self._embedding_model_name,
            vision=self._vision,
            dpi=self._dpi,
            source_type_weights=self._source_type_weights,
            on_ingestion_complete=self._on_ingestion_complete,
            lm_client=self._lm_client,
            graph_store=self._graph_store,
            ingestion_methods=self._delegate_methods,
            analyze_text_skip_threshold_chars=self._analyze_text_skip_threshold_chars,
            analyze_concurrency=self._analyze_concurrency,
            graph_config=self._graph_config,
        )

    def _service_ref(self) -> AnalyzedIngestionService:
        if self._service is None:
            self._service = self._build_service()
        return self._service

    async def analyze(self, **kwargs: Any) -> Source:
        return await self._service_ref().analyze(**kwargs)

    async def synthesize(self, source_id: str) -> Source:
        return await self._service_ref().synthesize(source_id)

    async def ingest(self, source_id: str) -> Source:
        return await self._service_ref().ingest(source_id)

    async def delete(self, source_id: str) -> None:
        await self._store.delete({"source_id": source_id})
