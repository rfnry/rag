from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from rfnry_knowledge.config.drawing import DrawingIngestionConfig
from rfnry_knowledge.ingestion.drawing.service import DrawingIngestionService
from rfnry_knowledge.ingestion.embeddings.base import BaseEmbeddings
from rfnry_knowledge.ingestion.vision.base import BaseVision
from rfnry_knowledge.models import Source
from rfnry_knowledge.providers import LLMClient
from rfnry_knowledge.stores.graph.base import BaseGraphStore
from rfnry_knowledge.stores.metadata.base import BaseMetadataStore
from rfnry_knowledge.stores.vector.base import BaseVectorStore


class DrawingIngestion:
    """Method wrapper around DrawingIngestionService."""

    required: bool = True

    def __init__(
        self,
        config: DrawingIngestionConfig,
        store: BaseVectorStore,
        embeddings: BaseEmbeddings,
        vision: BaseVision | None = None,
        lm_client: LLMClient | None = None,
        graph_store: BaseGraphStore | None = None,
        metadata_store: BaseMetadataStore | None = None,
        embedding_model_name: str = "",
        delegate_methods: list[Any] | None = None,
    ) -> None:
        if config.lm_client is None and lm_client is not None:
            config = replace(config, lm_client=lm_client)
        self._config = config
        self._store = store
        if not embedding_model_name:
            embedding_model_name = getattr(embeddings, "name", "") or ""
        self._embeddings = embeddings
        self._vision = vision
        self._lm_client = lm_client
        self._graph_store = graph_store
        self._metadata_store = metadata_store
        self._embedding_model_name = embedding_model_name
        self._delegate_methods = list(delegate_methods or [])
        self._service: DrawingIngestionService | None = None

    @property
    def name(self) -> str:
        return "drawing"

    def accepts(self, file_path: Path, source_type: str | None) -> bool:
        ext = file_path.suffix.lower()
        return ext == ".dxf" or (ext == ".pdf" and source_type == "drawing")

    def clone_for_store(self, store: BaseVectorStore) -> DrawingIngestion:
        return DrawingIngestion(
            config=self._config,
            store=store,
            embeddings=self._embeddings,
            vision=self._vision,
            lm_client=self._lm_client,
            graph_store=self._graph_store,
            metadata_store=self._metadata_store,
            embedding_model_name=self._embedding_model_name,
            delegate_methods=self._delegate_methods,
        )

    def bind(
        self,
        metadata_store: BaseMetadataStore,
        delegate_methods: list[Any],
    ) -> None:
        self._metadata_store = metadata_store
        self._delegate_methods = list(delegate_methods)
        self._service = None

    def _build_service(self) -> DrawingIngestionService:
        if self._metadata_store is None:
            raise RuntimeError("DrawingIngestion not bound — call bind() first")
        return DrawingIngestionService(
            config=self._config,
            embeddings=self._embeddings,
            vector_store=self._store,
            metadata_store=self._metadata_store,
            embedding_model_name=self._embedding_model_name,
            graph_store=self._graph_store,
            ingestion_methods=self._delegate_methods,
        )

    def _service_ref(self) -> DrawingIngestionService:
        if self._service is None:
            self._service = self._build_service()
        return self._service

    async def render(self, **kwargs: Any) -> Source:
        return await self._service_ref().render(**kwargs)

    async def extract(self, source_id: str) -> Source:
        return await self._service_ref().extract(source_id)

    async def link(self, source_id: str) -> Source:
        return await self._service_ref().link(source_id)

    async def ingest(self, source_id: str) -> Source:
        return await self._service_ref().ingest(source_id)

    async def delete(self, source_id: str) -> None:
        await self._store.delete({"source_id": source_id})
