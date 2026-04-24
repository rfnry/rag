"""DrawingIngestionService - sibling of AnalyzedIngestionService for drawing-first docs.

4 phases:
- render:  produce page images (PDF: rasterise; DXF: render to PNG via ezdxf)
- extract: per-page ``DrawingPageAnalysis`` via ``AnalyzeDrawingPage`` (PDF) or
           direct DXF parse
- link:    cross-sheet resolution (deterministic first, LLM residue)
- ingest:  embed + graph write

Phase bodies are ``NotImplementedError`` stubs for C3 and are filled in across
C4-C10. The service intentionally lives outside ``MethodNamespace`` so it can
own the 4-phase lifecycle the way ``AnalyzedIngestionService`` owns its 3
phases.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from baml_py import ClientRegistry

from rfnry_rag.retrieval.common.language_model import build_registry
from rfnry_rag.retrieval.common.logging import get_logger
from rfnry_rag.retrieval.common.models import Source
from rfnry_rag.retrieval.modules.ingestion.drawing.config import DrawingIngestionConfig
from rfnry_rag.retrieval.modules.ingestion.embeddings.base import BaseEmbeddings
from rfnry_rag.retrieval.stores.graph.base import BaseGraphStore
from rfnry_rag.retrieval.stores.metadata.base import BaseMetadataStore
from rfnry_rag.retrieval.stores.vector.base import BaseVectorStore

logger = get_logger("drawing/ingestion")

SUPPORTED_DRAWING_EXTENSIONS: frozenset[str] = frozenset({".dxf", ".pdf"})


class DrawingIngestionService:
    def __init__(
        self,
        config: DrawingIngestionConfig,
        embeddings: BaseEmbeddings,
        vector_store: BaseVectorStore,
        metadata_store: BaseMetadataStore,
        embedding_model_name: str,
        graph_store: BaseGraphStore | None = None,
        ingestion_methods: list[Any] | None = None,
    ) -> None:
        self._config = config
        self._embeddings = embeddings
        self._vector_store = vector_store
        self._metadata_store = metadata_store
        self._embedding_model_name = embedding_model_name
        self._graph_store = graph_store
        self._ingestion_methods = ingestion_methods or []
        self._registry: ClientRegistry | None = (
            build_registry(config.lm_client) if config.lm_client else None
        )

    async def render(
        self,
        file_path: str | Path,
        knowledge_id: str | None = None,
        source_type: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Source:
        """Phase 1 - produce page images. Idempotent on re-entry for the same file_hash. (C4)"""
        raise NotImplementedError

    async def extract(self, source_id: str) -> Source:
        """Phase 2 - per-page DrawingPageAnalysis. Idempotent when status != 'rendered'. (C5/C6)"""
        raise NotImplementedError

    async def link(self, source_id: str) -> Source:
        """Phase 3 - cross-sheet linking (deterministic + LLM residue). Idempotent. (C7/C8)"""
        raise NotImplementedError

    async def ingest(self, source_id: str) -> Source:
        """Phase 4 - embed + graph write. Idempotent. (C10)"""
        raise NotImplementedError
