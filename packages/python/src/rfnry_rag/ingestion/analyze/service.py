import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from baml_py import ClientRegistry
from baml_py import errors as baml_errors

from rfnry_rag.config.graph import GraphIngestionConfig
from rfnry_rag.exceptions import ConfigurationError, IngestionError
from rfnry_rag.ingestion.analyze.models import (
    CrossReference,
    DiscoveredEntity,
    DiscoveredTable,
    DocumentSynthesis,
    PageAnalysis,
    PageCluster,
)
from rfnry_rag.ingestion.analyze.parsers.l5x import parse_l5x
from rfnry_rag.ingestion.analyze.parsers.xml import is_l5x, parse_xml
from rfnry_rag.ingestion.analyze.pdf_splitter import iter_pdf_page_images
from rfnry_rag.ingestion.embeddings.base import BaseEmbeddings
from rfnry_rag.ingestion.embeddings.batching import embed_batched
from rfnry_rag.ingestion.hashing import file_hash as compute_file_hash
from rfnry_rag.ingestion.page_range import parse_page_range
from rfnry_rag.ingestion.vision.base import BaseVision
from rfnry_rag.logging import get_logger
from rfnry_rag.models import Source, VectorPoint
from rfnry_rag.providers import LanguageModelClient, build_registry
from rfnry_rag.stores.graph.base import BaseGraphStore
from rfnry_rag.stores.graph.mapper import cross_refs_to_graph_relations, page_entities_to_graph
from rfnry_rag.stores.metadata.base import BaseMetadataStore
from rfnry_rag.stores.vector.base import BaseVectorStore

logger = get_logger("analyze/ingestion")

STRUCTURED_EXTENSIONS = {".pdf", ".xml", ".l5x"}
_MAX_PAGES_PER_ENTITY = 20  # caps pairwise cross-ref expansion to 190 refs per entity (20*19/2)


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
        analyze_text_skip_threshold_chars: int = 300,
        analyze_concurrency: int = 5,
        graph_config: GraphIngestionConfig | None = None,
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
        self._analyze_text_skip_threshold_chars = analyze_text_skip_threshold_chars
        self._analyze_concurrency = analyze_concurrency
        self._graph_config = graph_config if graph_config is not None else GraphIngestionConfig()

    async def analyze(
        self,
        file_path: str | Path,
        knowledge_id: str | None = None,
        source_type: str | None = None,
        metadata: dict[str, Any] | None = None,
        page_range: str | None = None,
    ) -> Source:
        """Analyze phase: per-page/entity analysis. Returns Source with status='analyzed'."""
        file_path = Path(file_path)
        ext = file_path.suffix.lower()
        logger.info("[analyze/ingestion/analyze] processing file: %s (%s)", file_path.name, ext)

        # File-hash short-circuit: if an identical file was already analyzed for
        # this knowledge, return the existing Source without re-running any LLM calls.
        file_hash_value = await asyncio.to_thread(compute_file_hash, file_path)
        existing = await self._metadata_store.find_by_hash(file_hash_value, knowledge_id)
        if existing is not None and existing.status in ("analyzed", "synthesized", "completed"):
            logger.info(
                "[analyze] file-hash cache hit source=%s status=%s",
                existing.source_id,
                existing.status,
            )
            return existing

        page_filter = parse_page_range(page_range) if page_range else None
        notes: list[str] = []

        if ext == ".pdf":
            page_analyses = await self._analyze_pdf_with_cache(
                file_path, page_filter, knowledge_id, notes=notes
            )
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

        source_metadata: dict[str, Any] = {
            **(metadata or {}),
            "file_type": file_type,
            "file_name": file_path.name,
        }
        if notes:
            source_metadata["ingestion_notes"] = list(notes)

        source = Source(
            source_id=source_id,
            knowledge_id=knowledge_id,
            source_type=source_type,
            status="analyzed",
            embedding_model=self._embedding_model_name,
            file_hash=file_hash_value,
            created_at=datetime.now(UTC),
            source_weight=self._source_type_weights.get(source_type, 1.0) if source_type else 1.0,
            metadata=source_metadata,
        )
        await self._metadata_store.create_source(source)
        await self._metadata_store.upsert_page_analyses(
            source_id,
            [{"page_number": pa.page_number, "data": _serialize_analysis(pa)} for pa in page_analyses],
        )

        logger.info(
            "[analyze/ingestion/analyze] complete: %d pages/groups, status=analyzed",
            len(page_analyses),
        )
        return source

    async def synthesize(self, source_id: str) -> Source:
        """Synthesize phase: cross-page/entity relationship discovery. Returns Source with status='synthesized'."""
        source = await self._metadata_store.get_source(source_id)
        if source is None:
            raise IngestionError(f"source not found: {source_id}")
        if source.status in ("synthesized", "completed"):
            logger.info("[synthesize] source %s already %s, skipping", source_id, source.status)
            return source

        file_type = source.metadata.get("file_type")
        rows = await self._metadata_store.get_page_analyses(source_id)
        page_analyses = [_deserialize_analysis(r["data"]) for r in rows]

        logger.info("[analyze/ingestion/synthesize] source %s: %d pages in context", source_id, len(page_analyses))

        if file_type == "pdf":
            synthesis = await self._synthesize_pdf(page_analyses)
        elif file_type == "l5x":
            synthesis = self._synthesize_l5x(page_analyses)
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
        """Ingest phase: embed + store. Returns Source with status='completed'."""
        source = await self._metadata_store.get_source(source_id)
        if source is None:
            raise IngestionError(f"source not found: {source_id}")
        if source.status == "completed":
            logger.info("[ingest] source %s already completed, skipping", source_id)
            return source

        rows = await self._metadata_store.get_page_analyses(source_id)
        page_analyses = [_deserialize_analysis(r["data"]) for r in rows]
        synthesis_data = source.metadata.get("synthesis", {})
        synthesis = _deserialize_synthesis(synthesis_data)

        notes: list[str] = []

        logger.info("[analyze/ingestion/ingest] building multi-vector set for %d pages", len(page_analyses))

        xref_map: dict[int, list[int]] = {}
        for xref in synthesis.cross_references:
            xref_map.setdefault(xref.source_page, []).append(xref.target_page)
            xref_map.setdefault(xref.target_page, []).append(xref.source_page)

        texts_to_embed: list[str] = []
        payloads: list[dict[str, Any]] = []

        for pa in page_analyses:
            # 1. Description vector (always present)
            desc_parts = [pa.description]
            if pa.entities:
                desc_parts.append("Entities: " + ", ".join(e.name for e in pa.entities))
            if pa.annotations:
                desc_parts.append("Annotations: " + ", ".join(pa.annotations))
            desc_text = "\n".join(desc_parts)
            texts_to_embed.append(desc_text)
            payloads.append(
                _build_payload(
                    vector_role="description",
                    content=pa.description,
                    pa=pa,
                    source=source,
                    xref_map=xref_map,
                )
            )

            # 2. Raw-text vector (only when non-empty — skip silently for scanned/image pages)
            if pa.raw_text.strip():
                texts_to_embed.append(pa.raw_text)
                payloads.append(
                    _build_payload(
                        vector_role="raw_text",
                        content=pa.raw_text,
                        pa=pa,
                        source=source,
                        xref_map=xref_map,
                    )
                )

            # 3. Table-row vectors (one per row per table)
            for table in pa.tables:
                cols = table.columns or []
                for row in table.rows or []:
                    row_text = _format_table_row(table, cols, row)
                    texts_to_embed.append(row_text)
                    payloads.append(
                        _build_payload(
                            vector_role="table_row",
                            content=row_text,
                            pa=pa,
                            source=source,
                            xref_map=xref_map,
                            extra={"table_title": table.title or ""},
                        )
                    )

        # Single batched embed call for all texts
        vectors = await embed_batched(self._embeddings, texts_to_embed)
        if vectors:
            await self._vector_store.initialize(len(vectors[0]))

        points: list[VectorPoint] = [
            VectorPoint(point_id=str(uuid4()), vector=v, payload=p) for v, p in zip(vectors, payloads, strict=True)
        ]
        await self._vector_store.upsert(points)

        # Graph store — entities already extracted in phase 1, use mapper directly
        if self._graph_store:
            try:
                all_entities = []
                for pa in page_analyses:
                    all_entities.extend(page_entities_to_graph(pa, source.source_id, self._graph_config))
                relations = cross_refs_to_graph_relations(
                    synthesis,
                    page_analyses,
                    source.knowledge_id,
                    self._graph_config,
                )
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
                notes.append(f"graph:warn:method_failed({exc!s:.80})")

        # Delegate to other ingestion methods — prefer raw OCR text; fall back to enriched
        # description (with entity names) for L5X/XML where raw_text is always empty.
        full_text = "\n\n".join(_page_text_for_document_store(pa) for pa in page_analyses)
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
                    notes=notes,
                )
            except Exception as exc:
                logger.warning("[analyze/ingestion/ingest] method '%s' failed: %s", method.name, exc)
                notes.append(f"{method.name}:warn:method_failed({exc!s:.80})")

        if notes:
            source.metadata.setdefault("ingestion_notes", []).extend(notes)
            await self._metadata_store.update_source(
                source_id,
                status="completed",
                chunk_count=len(page_analyses),
                metadata=source.metadata,
            )
        else:
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

    async def _analyze_pdf_with_cache(
        self,
        file_path: Path,
        page_filter: set[int] | None,
        knowledge_id: str | None,
        notes: list[str] | None = None,
    ) -> list[PageAnalysis]:
        """PDF analysis with per-page image-hash caching and text-density pre-filter.

        Pages whose rendered-image hash matches an existing ``rag_page_analyses`` row
        are reused without an LLM call (cache hits). Of the remaining pages, those that
        have ``raw_text_char_count >= analyze_text_skip_threshold_chars`` AND no embedded
        images are built directly from raw text (``page_type='text'``) without a vision
        LLM call. The rest go through the BAML fan-out.

        Setting ``analyze_text_skip_threshold_chars=0`` disables the pre-filter entirely
        so all cache-miss pages go through vision — useful when every page must be
        visually inspected regardless of extractable text.
        """
        self._require_vision_and_registry()

        threshold = self._analyze_text_skip_threshold_chars
        pages = list(iter_pdf_page_images(file_path, dpi=self._dpi, pages=page_filter))
        # page_hash is empty string when the iterator doesn't provide it (e.g. in tests
        # that patch iter_pdf_page_images with minimal dicts). An empty hash is never
        # found in get_page_analyses_by_hash, so those pages fall through to the LLM.
        page_hashes = [p.get("page_hash", "") for p in pages]

        # Pass knowledge_id=None for a cross-knowledge page-hash lookup: the same
        # rendered-image bytes produce the same analysis regardless of which knowledge
        # they came from.  File-hash short-circuit (above in analyze()) is already
        # knowledge-scoped; per-page hash is content-addressed so cross-knowledge
        # reuse is safe.
        cached = await self._metadata_store.get_page_analyses_by_hash(
            page_hashes=page_hashes,
            knowledge_id=None,
        )

        # Classify cache-miss pages into text-only (no vision needed) and vision targets.
        text_only_pages: list[dict] = []
        vision_pages: list[dict] = []
        for p in pages:
            ph = p.get("page_hash", "")
            if ph and ph in cached:
                continue  # cache hit — handled below
            if threshold > 0 and p.get("raw_text_char_count", 0) >= threshold and not p.get("has_images", False):
                text_only_pages.append(p)
            else:
                vision_pages.append(p)

        logger.info(
            "[analyze] page-cache: %d hits / %d text-only / %d vision",
            len(pages) - len(text_only_pages) - len(vision_pages),
            len(text_only_pages),
            len(vision_pages),
        )

        # Run vision on targets that are not text-only.
        fresh_by_num: dict[int, PageAnalysis] = {}
        if vision_pages:
            sem = asyncio.Semaphore(self._analyze_concurrency)
            fresh_results = await asyncio.gather(
                *(self._analyze_one(p, sem, notes=notes) for p in vision_pages)
            )
            for f in fresh_results:
                if f is not None:
                    fresh_by_num[f.page_number] = f
            if not fresh_by_num and not text_only_pages:
                raise IngestionError(
                    "AnalyzePage failed on all pages — no document content extracted"
                )

        # Build PageAnalysis for text-only pages directly from raw text.
        for p in text_only_pages:
            raw = p.get("raw_text", "")
            fresh_by_num[p["page_number"]] = PageAnalysis(
                page_number=p["page_number"],
                description=raw[:5000],  # cap to protect against pathological pages
                entities=[],
                tables=[],
                annotations=[],
                page_type="text",
            )

        results: list[PageAnalysis] = []
        for p in pages:
            ph = p.get("page_hash", "")
            if ph and ph in cached:
                pa = _deserialize_analysis(cached[ph])
                pa.page_number = p["page_number"]
                pa.raw_text = p.get("raw_text", "")
                pa.page_hash = ph
            else:
                pa_or_none = fresh_by_num.get(p["page_number"])
                if pa_or_none is None:
                    continue
                pa = pa_or_none
                pa.raw_text = p.get("raw_text", "")
                pa.page_hash = ph
            results.append(pa)
        return results

    def _require_vision_and_registry(self) -> None:
        if not self._vision:
            raise ConfigurationError("vision provider required for structured PDF analysis")
        if not self._registry:
            raise ConfigurationError(
                "AnalyzedIngestion.lm_client is required for structured PDF analysis "
                "(used by AnalyzePage and SynthesizeDocument BAML functions). "
                "Provide a LanguageModelClient with your LLM provider and API key."
            )

    async def _analyze_one(
        self,
        img: dict,
        sem: asyncio.Semaphore,
        notes: list[str] | None = None,
    ) -> PageAnalysis | None:
        """Analyze a single page image via BAML AnalyzePage.

        Per-page failures soft-skip: returns None and appends a note to the
        caller-supplied list. The caller decides whether the surviving page
        count is enough to proceed.
        """
        self._require_vision_and_registry()

        from baml_py import Image

        from rfnry_rag.baml.baml_client.async_client import b

        registry: ClientRegistry = self._registry  # type: ignore[assignment]
        baml_image = Image.from_base64("image/png", img["image_base64"])
        page_number = img["page_number"]
        async with sem:
            try:
                result = await b.AnalyzePage(
                    baml_image,
                    baml_options={"client_registry": registry},
                )
            except baml_errors.BamlValidationError as exc:
                logger.warning("AnalyzePage invalid output on page %d: %s", page_number, exc)
                if notes is not None:
                    notes.append(f"vision:warn:page_{page_number}:invalid_output({exc!s:.80})")
                return None
            except Exception as exc:
                logger.warning("AnalyzePage failed on page %d: %s", page_number, exc)
                if notes is not None:
                    notes.append(f"vision:warn:page_{page_number}:{type(exc).__name__}({exc!s:.80})")
                return None

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

        logger.info(
            "[analyze/ingestion/analyze/vision] page %d analyzed (%s, %d entities)",
            img["page_number"],
            analysis.page_type,
            len(analysis.entities),
        )

        return analysis

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
                "AnalyzedIngestion.lm_client is required for PDF synthesis "
                "(used by SynthesizeDocument BAML function). "
                "Provide a LanguageModelClient with your LLM provider and API key."
            )

        from rfnry_rag.baml.baml_client.async_client import b

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
            if len(pages) > _MAX_PAGES_PER_ENTITY:
                logger.warning(
                    "entity %s appears on %d pages; capping cross-ref expansion to first %d",
                    entity_name,
                    len(pages),
                    _MAX_PAGES_PER_ENTITY,
                )
                pages = pages[:_MAX_PAGES_PER_ENTITY]
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

    def _synthesize_l5x(self, page_analyses: list[PageAnalysis]) -> DocumentSynthesis:
        """Deterministic cross-reference computation for L5X."""
        xrefs = self._synthesize_shared_entities(page_analyses)
        logger.info("[analyze/ingestion/synthesize/l5x] computed %d cross-references", len(xrefs))
        return DocumentSynthesis(cross_references=xrefs)

    def _synthesize_xml(self, page_analyses: list[PageAnalysis]) -> DocumentSynthesis:
        """Deterministic linking by shared entities for generic XML."""
        xrefs = self._synthesize_shared_entities(page_analyses)
        logger.info("[analyze/ingestion/synthesize/xml] computed %d cross-references", len(xrefs))
        return DocumentSynthesis(cross_references=xrefs)


def _page_text_for_document_store(pa: PageAnalysis) -> str:
    """Return the text to feed downstream document/BM25 methods.

    For PDF pages we prefer the raw OCR text (populated via PyMuPDF's
    ``page.get_text()``). For L5X / XML inputs raw_text is empty, so we
    fall back to the LLM description enriched with entity names and
    annotations — preserving the document-store searchability of entity
    tokens that the analysed pipeline surfaces for structured files.
    """
    if pa.raw_text:
        return pa.raw_text
    parts = [pa.description]
    if pa.entities:
        parts.append("Entities: " + ", ".join(e.name for e in pa.entities))
    if pa.annotations:
        parts.append("Annotations: " + ", ".join(pa.annotations))
    return "\n".join(parts)


def _build_payload(
    *,
    vector_role: str,
    content: str,
    pa: PageAnalysis,
    source: Source,
    xref_map: dict[int, list[int]],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a vector payload dict, tagged with ``vector_role`` for downstream filtering."""
    payload: dict[str, Any] = {
        "content": content,
        "vector_role": vector_role,
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
    }
    if extra:
        payload.update(extra)
    return payload


def _format_table_row(table: DiscoveredTable, cols: list[str], row: dict[str, str] | list[str]) -> str:
    """Serialise one table row as a string with column-header context.

    Handles both ``dict`` rows (from BAML ``map<string,string>[]``) and
    ``list`` rows, as well as absent column lists.  The table title is
    prefixed when present so the vector carries its own column context."""
    prefix = f"[{table.title}] " if table.title else ""
    if isinstance(row, dict):
        # Dict rows: emit key:value pairs preserving column order where possible
        pairs = [f"{c}: {row.get(c) or ''}" for c in cols] if cols else [f"{k}: {v or ''}" for k, v in row.items()]
        return prefix + " | ".join(pairs)
    # List rows: zip with column headers
    if not cols:
        return prefix + " | ".join(str(cell) for cell in row)
    pairs = []
    for i, cell in enumerate(row):
        if i < len(cols):
            pairs.append(f"{cols[i]}: {cell}")
        else:
            pairs.append(str(cell))
    return prefix + " | ".join(pairs)


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
        "raw_text": pa.raw_text,
        "page_hash": pa.page_hash,
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
        raw_text=data.get("raw_text", ""),
        page_hash=data.get("page_hash", ""),
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
