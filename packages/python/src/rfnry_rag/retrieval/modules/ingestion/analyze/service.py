import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from baml_py import ClientRegistry
from baml_py import errors as baml_errors

from rfnry_rag.retrieval.common.errors import ConfigurationError, IngestionError
from rfnry_rag.retrieval.common.hashing import file_hash as compute_file_hash
from rfnry_rag.retrieval.common.language_model import LanguageModelClient, build_registry
from rfnry_rag.retrieval.common.logging import get_logger
from rfnry_rag.retrieval.common.models import Source, VectorPoint
from rfnry_rag.retrieval.common.page_range import parse_page_range
from rfnry_rag.retrieval.modules.ingestion.analyze.models import (
    CrossReference,
    DiscoveredEntity,
    DiscoveredTable,
    DocumentSynthesis,
    PageAnalysis,
    PageCluster,
)
from rfnry_rag.retrieval.modules.ingestion.analyze.parsers.l5x import parse_l5x
from rfnry_rag.retrieval.modules.ingestion.analyze.parsers.xml import is_l5x, parse_xml
from rfnry_rag.retrieval.modules.ingestion.analyze.pdf_splitter import iter_pdf_page_images
from rfnry_rag.retrieval.modules.ingestion.embeddings.base import BaseEmbeddings
from rfnry_rag.retrieval.modules.ingestion.embeddings.utils import embed_batched
from rfnry_rag.retrieval.modules.ingestion.vision.base import BaseVision
from rfnry_rag.retrieval.stores.graph.base import BaseGraphStore
from rfnry_rag.retrieval.stores.graph.mapper import cross_refs_to_graph_relations, page_entities_to_graph
from rfnry_rag.retrieval.stores.metadata.base import BaseMetadataStore
from rfnry_rag.retrieval.stores.vector.base import BaseVectorStore

logger = get_logger("analyze/ingestion")

STRUCTURED_EXTENSIONS = {".pdf", ".xml", ".l5x"}


class AnalyzedIngestionService:
    def __init__(
        self,
        embeddings: BaseEmbeddings,
        vector_store: BaseVectorStore,
        metadata_store: BaseMetadataStore,
        embedding_model_name: str,
        vision: BaseVision | None = None,
        dpi: int = 300,
        source_type_weights: dict[str, float] | None = None,
        on_ingestion_complete: Callable[[str | None], Awaitable[None]] | None = None,
        lm_client: LanguageModelClient | None = None,
        graph_store: BaseGraphStore | None = None,
        ingestion_methods: list | None = None,
    ) -> None:
        self._embeddings = embeddings
        self._vector_store = vector_store
        self._metadata_store = metadata_store
        self._embedding_model_name = embedding_model_name
        self._vision = vision
        self._dpi = dpi
        self._source_type_weights = source_type_weights or {}
        self._on_ingestion_complete = on_ingestion_complete
        self._registry: ClientRegistry | None = build_registry(lm_client) if lm_client else None
        self._graph_store = graph_store
        self._ingestion_methods = ingestion_methods or []

    async def analyze(
        self,
        file_path: str | Path,
        knowledge_id: str | None = None,
        source_type: str | None = None,
        metadata: dict[str, Any] | None = None,
        page_range: str | None = None,
    ) -> Source:
        """Phase 1: Per-page/entity analysis. Returns Source with status='analyzed'."""
        file_path = Path(file_path)
        ext = file_path.suffix.lower()
        logger.info("[analyze/ingestion/analyze] processing file: %s (%s)", file_path.name, ext)

        page_filter = parse_page_range(page_range) if page_range else None

        if ext == ".pdf":
            page_analyses = await self._analyze_pdf(file_path, page_filter)
            file_type = "pdf"
        elif ext == ".l5x" or (ext == ".xml" and is_l5x(file_path)):
            page_analyses = self._analyze_l5x(file_path)
            file_type = "l5x"
        elif ext == ".xml":
            page_analyses = parse_xml(file_path)
            file_type = "xml"
        else:
            raise IngestionError(f"unsupported structured file extension: {ext}")

        source_id = str(uuid4())
        # File hashing is CPU/IO-bound and must run off the event loop so a
        # large PDF doesn't block every other coroutine. chunk/service.py:148
        # already does this for the unstructured path — analyze must match.
        file_hash_value = await asyncio.to_thread(compute_file_hash, file_path)

        source = Source(
            source_id=source_id,
            knowledge_id=knowledge_id,
            source_type=source_type,
            status="analyzed",
            embedding_model=self._embedding_model_name,
            file_hash=file_hash_value,
            created_at=datetime.now(UTC),
            source_weight=self._source_type_weights.get(source_type, 1.0) if source_type else 1.0,
            metadata={
                **(metadata or {}),
                "file_type": file_type,
                "file_name": file_path.name,
                "page_analyses": [_serialize_analysis(a) for a in page_analyses],
            },
        )
        await self._metadata_store.create_source(source)

        logger.info(
            "[analyze/ingestion/analyze] complete: %d pages/groups, status=analyzed",
            len(page_analyses),
        )
        return source

    async def synthesize(self, source_id: str) -> Source:
        """Phase 2: Cross-page/entity relationship discovery. Returns Source with status='synthesized'."""
        source = await self._metadata_store.get_source(source_id)
        if source is None:
            raise IngestionError(f"source not found: {source_id}")

        file_type = source.metadata.get("file_type")
        page_analyses = [_deserialize_analysis(a) for a in source.metadata["page_analyses"]]

        logger.info("[analyze/ingestion/synthesize] source %s: %d pages in context", source_id, len(page_analyses))

        if file_type == "pdf":
            synthesis = await self._synthesize_pdf(page_analyses)
        elif file_type == "l5x":
            synthesis = self._synthesize_l5x(source)
        else:
            synthesis = self._synthesize_xml(page_analyses)

        await self._metadata_store.update_source(
            source_id,
            status="synthesized",
            metadata={**source.metadata, "synthesis": _serialize_synthesis(synthesis)},
        )

        source.status = "synthesized"
        logger.info("[analyze/ingestion/synthesize] complete: status=synthesized")
        return source

    async def ingest(self, source_id: str) -> Source:
        """Phase 3: Embed + store. Returns Source with status='completed'."""
        source = await self._metadata_store.get_source(source_id)
        if source is None:
            raise IngestionError(f"source not found: {source_id}")

        page_analyses = [_deserialize_analysis(a) for a in source.metadata["page_analyses"]]
        synthesis_data = source.metadata.get("synthesis", {})
        synthesis = _deserialize_synthesis(synthesis_data)

        logger.info("[analyze/ingestion/ingest] embedding %d page descriptions", len(page_analyses))

        texts = []
        for pa in page_analyses:
            parts = [pa.description]
            if pa.entities:
                parts.append("Entities: " + ", ".join(e.name for e in pa.entities))
            if pa.annotations:
                parts.append("Annotations: " + ", ".join(pa.annotations))
            texts.append("\n".join(parts))

        vectors = await embed_batched(self._embeddings, texts)
        if vectors:
            await self._vector_store.initialize(len(vectors[0]))

        xref_map: dict[int, list[int]] = {}
        for xref in synthesis.cross_references:
            xref_map.setdefault(xref.source_page, []).append(xref.target_page)
            xref_map.setdefault(xref.target_page, []).append(xref.source_page)

        points: list[VectorPoint] = []
        for pa, vector in zip(page_analyses, vectors, strict=True):
            points.append(
                VectorPoint(
                    point_id=str(uuid4()),
                    vector=vector,
                    payload={
                        "content": pa.description,
                        "source_type": "structured",
                        "file_type": source.metadata.get("file_type", ""),
                        "page_number": pa.page_number,
                        "page_type": pa.page_type,
                        "entities": [e.name for e in pa.entities],
                        "entity_categories": [e.category for e in pa.entities],
                        "cross_references": sorted(set(xref_map.get(pa.page_number, []))),
                        "source_id": source.source_id,
                        "knowledge_id": source.knowledge_id,
                        "source_name": source.metadata.get("file_name", ""),
                    },
                )
            )

        await self._vector_store.upsert(points)

        # Graph store — entities already extracted in phase 1, use mapper directly
        if self._graph_store:
            try:
                all_entities = []
                for pa in page_analyses:
                    all_entities.extend(page_entities_to_graph(pa, source.source_id))
                relations = cross_refs_to_graph_relations(synthesis, page_analyses, source.knowledge_id)
                if all_entities:
                    await self._graph_store.add_entities(
                        source_id=source.source_id,
                        knowledge_id=source.knowledge_id,
                        entities=all_entities,
                    )
                if relations:
                    await self._graph_store.add_relations(
                        source_id=source.source_id,
                        relations=relations,
                    )
                logger.info(
                    "[analyze/ingestion/ingest] graph: %d entities, %d relations",
                    len(all_entities),
                    len(relations),
                )
            except Exception as exc:
                logger.warning("[analyze/ingestion/ingest] graph failed: %s", exc)

        # Delegate to other ingestion methods (document, etc.)
        full_text = "\n\n".join(texts)
        for method in self._ingestion_methods:
            try:
                await method.ingest(
                    source_id=source.source_id,
                    knowledge_id=source.knowledge_id,
                    source_type=source.source_type,
                    source_weight=source.source_weight,
                    title=source.metadata.get("file_name", ""),
                    full_text=full_text,
                    chunks=[],
                    tags=[],
                    metadata=source.metadata,
                )
            except Exception as exc:
                logger.warning("[analyze/ingestion/ingest] method '%s' failed: %s", method.name, exc)

        await self._metadata_store.update_source(
            source_id,
            status="completed",
            chunk_count=len(page_analyses),
        )

        source.status = "completed"

        if self._on_ingestion_complete:
            await self._on_ingestion_complete(source.knowledge_id)

        logger.info("[analyze/ingestion/ingest] complete: %d vectors, status=completed", len(points))
        return source

    async def _analyze_pdf(self, file_path: Path, page_range: set[int] | None) -> list[PageAnalysis]:
        if not self._vision:
            raise ConfigurationError("vision provider required for structured PDF analysis")
        if not self._registry:
            raise ConfigurationError(
                "IngestionConfig.lm_client is required for structured PDF analysis "
                "(used by AnalyzePage and SynthesizeDocument BAML functions). "
                "Provide a LanguageModelClient with your LLM provider and API key."
            )

        analyses: list[PageAnalysis] = []

        from rfnry_rag.retrieval.baml.baml_client.async_client import b

        for img in iter_pdf_page_images(file_path, dpi=self._dpi, pages=page_range):
            from baml_py import Image

            baml_image = Image.from_base64("image/png", img["image_base64"])
            try:
                result = await b.AnalyzePage(
                    baml_image,
                    baml_options={"client_registry": self._registry},
                )
            except baml_errors.BamlValidationError as exc:
                raise IngestionError(
                    f"AnalyzePage failed on page {img['page_number']}: LLM returned an unparseable response. "
                    f"This usually means the model does not support the expected output format. Detail: {exc}"
                ) from exc
            except Exception as exc:
                raise IngestionError(f"AnalyzePage failed on page {img['page_number']}: {exc}") from exc

            analysis = PageAnalysis(
                page_number=img["page_number"],
                description=result.description,
                entities=[
                    DiscoveredEntity(name=e.name, category=e.category, context=e.context, value=e.value)
                    for e in result.entities
                ],
                tables=[DiscoveredTable(title=t.title, columns=t.columns, rows=t.rows) for t in result.tables],
                annotations=result.annotations,
                page_type=result.page_type,
            )
            analyses.append(analysis)

            logger.info(
                "[analyze/ingestion/analyze/vision] page %d analyzed (%s, %d entities)",
                img["page_number"],
                analysis.page_type,
                len(analysis.entities),
            )

        return analyses

    def _analyze_l5x(self, file_path: Path) -> list[PageAnalysis]:
        docs = parse_l5x(file_path)
        logger.info("[analyze/ingestion/analyze/l5x] extracted %d entity groups", len(docs))
        return [
            PageAnalysis(
                page_number=i + 1,
                description=doc.content,
                page_type=doc.doc_type,
                metadata={"doc_name": doc.name, "doc_path": doc.path},
            )
            for i, doc in enumerate(docs)
        ]

    async def _synthesize_pdf(self, page_analyses: list[PageAnalysis]) -> DocumentSynthesis:
        """Use LLM to find cross-page relationships in PDF analyses."""
        if not self._registry:
            raise ConfigurationError(
                "IngestionConfig.lm_client is required for PDF synthesis "
                "(used by SynthesizeDocument BAML function). "
                "Provide a LanguageModelClient with your LLM provider and API key."
            )

        from rfnry_rag.retrieval.baml.baml_client.async_client import b

        context_parts = []
        for pa in page_analyses:
            parts = [f"Page {pa.page_number} ({pa.page_type}): {pa.description}"]
            if pa.entities:
                parts.append(f"  Entities: {', '.join(e.name for e in pa.entities)}")
            context_parts.append("\n".join(parts))

        context = "\n\n".join(context_parts)

        try:
            result = await b.SynthesizeDocument(
                context,
                baml_options={"client_registry": self._registry},
            )
        except baml_errors.BamlValidationError as exc:
            raise IngestionError(
                f"SynthesizeDocument failed: LLM returned an unparseable response. "
                f"This usually means the model does not support the expected output format. Detail: {exc}"
            ) from exc
        except Exception as exc:
            raise IngestionError(f"SynthesizeDocument failed: {exc}") from exc

        synthesis = DocumentSynthesis(
            cross_references=[
                CrossReference(
                    source_page=xr.source_page,
                    target_page=xr.target_page,
                    relationship=xr.relationship,
                    shared_entities=xr.shared_entities,
                )
                for xr in result.cross_references
            ],
            page_clusters=[PageCluster(pages=pc.pages, reason=pc.reason) for pc in result.page_clusters],
            document_summary=result.document_summary,
        )

        logger.info(
            "[analyze/ingestion/synthesize/vision] found %d cross-references, %d page clusters",
            len(synthesis.cross_references),
            len(synthesis.page_clusters),
        )
        return synthesis

    @staticmethod
    def _synthesize_shared_entities(page_analyses: list[PageAnalysis]) -> list[CrossReference]:
        """Build cross-references from shared entities across pages.

        Both L5X and XML synthesis use identical logic: collect the pages on
        which each entity name appears, then emit a pairwise ``CrossReference``
        for every pair of pages that share the same entity.
        """
        xrefs: list[CrossReference] = []
        entity_pages: dict[str, list[int]] = {}

        for pa in page_analyses:
            for entity in pa.entities:
                entity_pages.setdefault(entity.name, []).append(pa.page_number)

        for entity_name, pages in entity_pages.items():
            if len(pages) > 1:
                for i, src_page in enumerate(pages):
                    for tgt_page in pages[i + 1 :]:
                        xrefs.append(
                            CrossReference(
                                source_page=src_page,
                                target_page=tgt_page,
                                relationship=f"shared entity: {entity_name}",
                                shared_entities=[entity_name],
                            )
                        )

        return xrefs

    def _synthesize_l5x(self, source: Source) -> DocumentSynthesis:
        """Deterministic cross-reference computation for L5X."""
        page_analyses = [_deserialize_analysis(a) for a in source.metadata["page_analyses"]]
        xrefs = self._synthesize_shared_entities(page_analyses)
        logger.info("[analyze/ingestion/synthesize/l5x] computed %d cross-references", len(xrefs))
        return DocumentSynthesis(cross_references=xrefs)

    def _synthesize_xml(self, page_analyses: list[PageAnalysis]) -> DocumentSynthesis:
        """Deterministic linking by shared entities for generic XML."""
        xrefs = self._synthesize_shared_entities(page_analyses)
        logger.info("[analyze/ingestion/synthesize/xml] computed %d cross-references", len(xrefs))
        return DocumentSynthesis(cross_references=xrefs)


def _serialize_analysis(pa: PageAnalysis) -> dict[str, Any]:
    return {
        "page_number": pa.page_number,
        "description": pa.description,
        "entities": [
            {"name": e.name, "category": e.category, "context": e.context, "value": e.value} for e in pa.entities
        ],
        "tables": [{"title": t.title, "columns": t.columns, "rows": t.rows} for t in pa.tables],
        "annotations": pa.annotations,
        "page_type": pa.page_type,
        "metadata": pa.metadata,
    }


def _deserialize_analysis(data: dict[str, Any]) -> PageAnalysis:
    return PageAnalysis(
        page_number=data["page_number"],
        description=data["description"],
        entities=[
            DiscoveredEntity(
                name=e["name"],
                category=e["category"],
                context=e.get("context", ""),
                value=e.get("value"),
            )
            for e in data.get("entities", [])
        ],
        tables=[
            DiscoveredTable(
                title=t.get("title"),
                columns=t.get("columns", []),
                rows=t.get("rows", []),
            )
            for t in data.get("tables", [])
        ],
        annotations=data.get("annotations", []),
        page_type=data.get("page_type", ""),
        metadata=data.get("metadata", {}),
    )


def _serialize_synthesis(synthesis: DocumentSynthesis) -> dict[str, Any]:
    return {
        "cross_references": [
            {
                "source_page": xr.source_page,
                "target_page": xr.target_page,
                "relationship": xr.relationship,
                "shared_entities": xr.shared_entities,
            }
            for xr in synthesis.cross_references
        ],
        "page_clusters": [{"pages": pc.pages, "reason": pc.reason} for pc in synthesis.page_clusters],
        "document_summary": synthesis.document_summary,
    }


def _deserialize_synthesis(data: dict[str, Any]) -> DocumentSynthesis:
    if not data:
        return DocumentSynthesis()
    return DocumentSynthesis(
        cross_references=[
            CrossReference(
                source_page=xr["source_page"],
                target_page=xr["target_page"],
                relationship=xr["relationship"],
                shared_entities=xr.get("shared_entities", []),
            )
            for xr in data.get("cross_references", [])
        ],
        page_clusters=[PageCluster(pages=pc["pages"], reason=pc["reason"]) for pc in data.get("page_clusters", [])],
        document_summary=data.get("document_summary", ""),
    )
