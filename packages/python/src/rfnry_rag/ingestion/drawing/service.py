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

from rfnry_rag.common.language_model import build_registry
from rfnry_rag.common.logging import get_logger
from rfnry_rag.ingestion.drawing.config import DrawingIngestionConfig
from rfnry_rag.ingestion.drawing.extract_dxf import extract_dxf_analysis
from rfnry_rag.ingestion.drawing.linker import pair_off_page_connectors
from rfnry_rag.ingestion.drawing.models import (
    DetectedComponent,
    DetectedConnection,
    DrawingPageAnalysis,
)
from rfnry_rag.ingestion.drawing.render import render_dxf, render_pdf_pages
from rfnry_rag.ingestion.embeddings.base import BaseEmbeddings
from rfnry_rag.ingestion.embeddings.utils import embed_batched
from rfnry_rag.retrieval.common.errors import IngestionError
from rfnry_rag.retrieval.common.hashing import file_hash as compute_file_hash
from rfnry_rag.retrieval.common.models import Source, VectorPoint
from rfnry_rag.stores.graph.base import BaseGraphStore
from rfnry_rag.stores.graph.drawing_mapper import drawing_to_graph
from rfnry_rag.stores.metadata.base import BaseMetadataStore
from rfnry_rag.stores.vector.base import BaseVectorStore

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
        self._registry: ClientRegistry | None = build_registry(config.lm_client) if config.lm_client else None

    async def render(
        self,
        file_path: str | Path,
        knowledge_id: str | None = None,
        source_type: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Source:
        """Render phase — produce page images. Idempotent on re-entry for the same file_hash."""
        path = Path(file_path)
        if not path.exists():
            raise IngestionError(f"drawing file not found: {path}")
        ext = path.suffix.lower()
        if ext not in SUPPORTED_DRAWING_EXTENSIONS:
            raise IngestionError(
                f"unsupported drawing extension: {ext!r}. Supported: {sorted(SUPPORTED_DRAWING_EXTENSIONS)}"
            )

        file_hash_value = await asyncio.to_thread(compute_file_hash, path)

        existing = await self._metadata_store.find_by_hash(file_hash_value, knowledge_id)
        if existing is not None and existing.status in {"rendered", "extracted", "linked", "completed"}:
            return existing

        if ext == ".pdf":
            pages = await asyncio.to_thread(lambda: list(render_pdf_pages(path, dpi=self._config.dpi)))
            source_format = "pdf"
        else:  # .dxf
            pages = await asyncio.to_thread(render_dxf, path, self._config.dpi)
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
        """Extract phase — per-page DrawingPageAnalysis. Idempotent when status != 'rendered'."""
        from rfnry_rag.ingestion.drawing.extract_pdf import (
            extract_pdf_analyses,
        )

        source = await self._metadata_store.get_source(source_id)
        if source is None:
            raise IngestionError(f"source not found: {source_id}")
        if source.status in {"extracted", "linked", "completed"}:
            return source  # idempotent
        if source.status != "rendered":
            raise IngestionError(f"extract requires status='rendered', got {source.status!r}")

        page_rows = await self._metadata_store.get_page_analyses(source_id)
        source_format = source.metadata.get("source_format")

        if source_format == "pdf":
            if self._registry is None:
                raise IngestionError("DrawingIngestionConfig.lm_client is required for PDF extract phase")
            analyses = await extract_pdf_analyses(
                page_rows,
                self._config,
                self._registry,
                source.metadata,
            )
        elif source_format == "dxf":
            file_path = Path(source.metadata["file_path"])
            analyses = await asyncio.to_thread(extract_dxf_analysis, file_path, self._config)
        else:
            raise IngestionError(f"unsupported source_format for extract: {source_format!r}")

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
        """Link phase — cross-sheet linking (deterministic + LLM residue). Idempotent."""
        source = await self._metadata_store.get_source(source_id)
        if source is None:
            raise IngestionError(f"source not found: {source_id}")
        if source.status in {"linked", "completed"}:
            return source  # idempotent
        if source.status != "extracted":
            raise IngestionError(f"link requires status='extracted', got {source.status!r}")

        rows = await self._metadata_store.get_page_analyses(source_id)
        pages: list[DrawingPageAnalysis] = []
        for r in rows:
            analysis_dict = r.get("data", {}).get("analysis")
            if analysis_dict is None:
                continue
            pages.append(DrawingPageAnalysis.from_dict(analysis_dict))

        deterministic_pairings = pair_off_page_connectors(pages)

        link_payload = {"deterministic_pairings": [p.to_dict() for p in deterministic_pairings]}
        new_metadata = {**source.metadata, "drawing_linking": link_payload}
        await self._metadata_store.update_source(
            source_id,
            status="linked",
            metadata=new_metadata,
        )
        source.status = "linked"
        source.metadata = new_metadata
        logger.info(
            "[drawing/link] source_id=%s deterministic=%d",
            source_id,
            len(deterministic_pairings),
        )
        return source

    async def ingest(self, source_id: str) -> Source:
        """Ingest phase — embed + graph write. Idempotent on completed."""
        source = await self._metadata_store.get_source(source_id)
        if source is None:
            raise IngestionError(f"source not found: {source_id}")
        if source.status == "completed":
            return source
        if source.status != "linked":
            raise IngestionError(f"ingest requires status='linked', got {source.status!r}")

        rows = await self._metadata_store.get_page_analyses(source_id)
        pages: list[DrawingPageAnalysis] = []
        for r in rows:
            analysis = r.get("data", {}).get("analysis")
            if analysis is None:
                continue
            pages.append(DrawingPageAnalysis.from_dict(analysis))

        linking = source.metadata.get("drawing_linking", {}) or {}
        deterministic_pairings = [DetectedConnection.from_dict(d) for d in linking.get("deterministic_pairings", [])]

        # Embed one vector per component (type + label + page-local neighbours + domain).
        component_texts: list[str] = []
        component_payloads: list[dict[str, Any]] = []
        for pa in pages:
            for c in pa.components:
                text = _describe_component(c, pa)
                component_texts.append(text)
                component_payloads.append(
                    {
                        "content": text,
                        "vector_role": "drawing_component",
                        "source_type": "drawing",
                        "source_id": source.source_id,
                        "knowledge_id": source.knowledge_id,
                        "component_id": c.component_id,
                        "symbol_class": c.symbol_class,
                        "page_number": pa.page_number,
                        "bbox": c.bbox,
                        "domain": pa.domain,
                        "source_name": source.metadata.get("file_name", ""),
                    }
                )

        if component_texts:
            vectors = await embed_batched(self._embeddings, component_texts)
            if vectors:
                await self._vector_store.initialize(len(vectors[0]))
                points = [
                    VectorPoint(point_id=str(uuid4()), vector=v, payload=p)
                    for v, p in zip(vectors, component_payloads, strict=True)
                ]
                await self._vector_store.upsert(points)

        # Graph writes (batched).
        relations_count = 0
        if self._graph_store is not None:
            entities, relations = drawing_to_graph(
                pages=pages,
                deterministic_pairings=deterministic_pairings,
                source_id=source.source_id,
                config=self._config,
                knowledge_id=source.knowledge_id,
            )
            relations_count = len(relations)
            if entities:
                await self._graph_store.add_entities(
                    source_id=source.source_id,
                    knowledge_id=source.knowledge_id,
                    entities=entities,
                )
            batch = self._config.graph_write_batch_size
            for i in range(0, len(relations), batch):
                await self._graph_store.add_relations(
                    source_id=source.source_id,
                    relations=relations[i : i + batch],
                )

        await self._metadata_store.update_source(
            source_id,
            status="completed",
            chunk_count=len(component_texts),
        )
        source.status = "completed"
        source.chunk_count = len(component_texts)
        batch_size = self._config.graph_write_batch_size
        relations_batches = 0 if self._graph_store is None else (relations_count + batch_size - 1) // batch_size
        logger.info(
            "[drawing/ingest] source_id=%s components=%d relations_batches=%d",
            source.source_id,
            len(component_texts),
            relations_batches,
        )
        return source


def _describe_component(c: DetectedComponent, pa: DrawingPageAnalysis) -> str:
    """Build an embedding-friendly description: type + label + same-page neighbours + domain."""
    parts: list[str] = [c.symbol_class]
    if c.label:
        parts.append(f"labelled {c.label}")
    neighbours: set[str] = set()
    for conn in pa.connections:
        if conn.from_component == c.component_id:
            neighbours.add(conn.to_component)
        elif conn.to_component == c.component_id:
            neighbours.add(conn.from_component)
    if neighbours:
        parts.append("connected to " + ", ".join(sorted(neighbours)))
    parts.append(f"on {pa.domain} drawing page {pa.page_number}")
    return " ".join(parts)
