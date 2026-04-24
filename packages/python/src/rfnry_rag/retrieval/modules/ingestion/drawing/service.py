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

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from baml_py import ClientRegistry

from rfnry_rag.retrieval.common.errors import IngestionError
from rfnry_rag.retrieval.common.hashing import file_hash as compute_file_hash
from rfnry_rag.retrieval.common.language_model import build_registry
from rfnry_rag.retrieval.common.logging import get_logger
from rfnry_rag.retrieval.common.models import Source
from rfnry_rag.retrieval.modules.ingestion.drawing.config import DrawingIngestionConfig
from rfnry_rag.retrieval.modules.ingestion.drawing.render import render_dxf, render_pdf_pages
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
        path = Path(file_path)
        if not path.exists():
            raise IngestionError(f"drawing file not found: {path}")
        ext = path.suffix.lower()
        if ext not in SUPPORTED_DRAWING_EXTENSIONS:
            raise IngestionError(
                f"unsupported drawing extension: {ext!r}. "
                f"Supported: {sorted(SUPPORTED_DRAWING_EXTENSIONS)}"
            )

        file_hash_value = await asyncio.to_thread(compute_file_hash, path)

        existing = await self._metadata_store.find_by_hash(file_hash_value, knowledge_id)
        if existing is not None and existing.status in {"rendered", "extracted", "linked", "completed"}:
            return existing

        if ext == ".pdf":
            pages = await asyncio.to_thread(
                lambda: list(render_pdf_pages(path, dpi=self._config.dpi))
            )
            source_format = "pdf"
        else:  # .dxf
            pages = [await asyncio.to_thread(render_dxf, path, self._config.dpi)]
            source_format = "dxf"

        source_id = str(uuid4())
        source = Source(
            source_id=source_id,
            knowledge_id=knowledge_id,
            source_type=source_type,
            status="rendered",
            embedding_model=self._embedding_model_name,
            file_hash=file_hash_value,
            created_at=datetime.now(UTC),
            metadata={
                **(metadata or {}),
                "source_format": source_format,
                "file_name": path.name,
                "file_path": str(path),
                "page_count": len(pages),
            },
        )
        await self._metadata_store.create_source(source)
        await self._metadata_store.upsert_page_analyses(
            source_id,
            [
                {
                    "page_number": p["page_number"],
                    "data": {
                        "page_image_b64": p["image_base64"],
                        "page_hash": p["page_hash"],
                        "raw_text": p.get("raw_text", ""),
                        "source_format": source_format,
                    },
                }
                for p in pages
            ],
        )
        logger.info(
            "[drawing/render] source_id=%s pages=%d format=%s",
            source_id,
            len(pages),
            source_format,
        )
        return source

    async def extract(self, source_id: str) -> Source:
        """Phase 2 - per-page DrawingPageAnalysis. Idempotent when status != 'rendered'. (C5/C6)"""
        from rfnry_rag.retrieval.modules.ingestion.drawing.extract_pdf import (
            extract_pdf_analyses,
        )

        source = await self._metadata_store.get_source(source_id)
        if source is None:
            raise IngestionError(f"source not found: {source_id}")
        if source.status in {"extracted", "linked", "completed"}:
            return source  # idempotent
        if source.status != "rendered":
            raise IngestionError(
                f"extract requires status='rendered', got {source.status!r}"
            )

        page_rows = await self._metadata_store.get_page_analyses(source_id)
        source_format = source.metadata.get("source_format")

        if source_format == "pdf":
            if self._registry is None:
                raise IngestionError(
                    "DrawingIngestionConfig.lm_client is required for PDF extract phase"
                )
            analyses = await extract_pdf_analyses(
                page_rows, self._config, self._registry, source.metadata,
            )
        elif source_format == "dxf":
            # C6 will fill this; for now leave a stub.
            raise NotImplementedError("DXF extract - see Task C6")
        else:
            raise IngestionError(
                f"unsupported source_format for extract: {source_format!r}"
            )

        # Production upsert_page_analyses REPLACES data_json wholesale. Merge at
        # the service layer so render-produced fields (page_image_b64, page_hash,
        # raw_text, source_format) are preserved alongside the new 'analysis'
        # payload. Keyed on page_number.
        existing_by_page: dict[int, dict] = {r["page_number"]: r for r in page_rows}
        merged: list[dict] = []
        for a in analyses:
            prior_data = existing_by_page.get(a.page_number, {}).get("data", {})
            merged.append(
                {
                    "page_number": a.page_number,
                    "data": {**prior_data, "analysis": a.to_dict()},
                }
            )
        await self._metadata_store.upsert_page_analyses(source_id, merged)
        await self._metadata_store.update_source(source_id, status="extracted")
        source.status = "extracted"
        logger.info(
            "[drawing/extract] source_id=%s pages=%d format=%s",
            source_id,
            len(analyses),
            source_format,
        )
        return source

    async def link(self, source_id: str) -> Source:
        """Phase 3 - cross-sheet linking (deterministic + LLM residue). Idempotent. (C7/C8)"""
        raise NotImplementedError

    async def ingest(self, source_id: str) -> Source:
        """Phase 4 - embed + graph write. Idempotent. (C10)"""
        raise NotImplementedError
