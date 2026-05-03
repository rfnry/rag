from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from rfnry_knowledge.common.logging import get_logger
from rfnry_knowledge.exceptions import (
    ConfigurationError,
    DuplicateSourceError,
    EmptyDocumentError,
    EnrichmentSkipped,
    IngestionError,
)
from rfnry_knowledge.ingestion.chunk.chunker import SemanticChunker
from rfnry_knowledge.ingestion.chunk.context import contextualize_chunks
from rfnry_knowledge.ingestion.chunk.contextualize import contextualize_chunks_with_llm
from rfnry_knowledge.ingestion.chunk.expand import expand_chunks
from rfnry_knowledge.ingestion.chunk.parsers.pdf import PDFParser
from rfnry_knowledge.ingestion.chunk.parsers.text import TextParser
from rfnry_knowledge.ingestion.hashing import file_hash
from rfnry_knowledge.ingestion.models import ParsedPage
from rfnry_knowledge.ingestion.notes import record_skip
from rfnry_knowledge.ingestion.page_range import parse_page_range
from rfnry_knowledge.ingestion.vision.base import BaseVision
from rfnry_knowledge.ingestion.vision.constants import IMAGE_EXTENSIONS
from rfnry_knowledge.models import Source
from rfnry_knowledge.providers.protocols import TokenCounter
from rfnry_knowledge.stores.metadata.base import BaseMetadataStore
from rfnry_knowledge.telemetry.context import set_ingest_field

if TYPE_CHECKING:
    from baml_py import ClientRegistry

    from rfnry_knowledge.config.ingestion import ContextualChunkConfig, DocumentExpansionConfig
    from rfnry_knowledge.ingestion.base import BaseIngestionMethod

logger = get_logger("chunk/ingestion")

INGESTION_BATCH_SIZE = 20

FILE_PARSERS_BY_EXTENSION: dict[str, type] = {
    ".pdf": PDFParser,
    ".txt": TextParser,
    ".md": TextParser,
    ".text": TextParser,
}

IngestionProgress = Callable[[int, int], Awaitable[None]]


class IngestionService:
    def __init__(
        self,
        chunker: SemanticChunker,
        ingestion_methods: list[BaseIngestionMethod],
        embedding_model_name: str = "",
        source_type_weights: dict[str, float] | None = None,
        metadata_store: BaseMetadataStore | None = None,
        on_ingestion_complete: Callable[[str | None], Awaitable[None]] | None = None,
        vision_parser: BaseVision | None = None,
        chunk_context_headers: bool = True,
        document_expansion: DocumentExpansionConfig | None = None,
        expansion_registry: ClientRegistry | None = None,
        contextual_chunk: ContextualChunkConfig | None = None,
        token_counter: TokenCounter | None = None,
    ) -> None:
        self._chunker = chunker
        self._ingestion_methods = ingestion_methods
        self._metadata_store = metadata_store
        self._embedding_model_name = embedding_model_name
        self._source_type_weights = source_type_weights
        self._on_ingestion_complete = on_ingestion_complete
        self._vision_parser = vision_parser
        self._chunk_context_headers = chunk_context_headers
        self._document_expansion = document_expansion
        self._expansion_registry = expansion_registry
        self._contextual_chunk = contextual_chunk
        self._token_counter = token_counter

    def _resolve_weight(self, source_type: str | None) -> float:
        if self._source_type_weights is None:
            return 1.0
        if source_type is None:
            return 1.0
        if source_type not in self._source_type_weights:
            raise ConfigurationError(
                f"source_type '{source_type}' is not defined in source_type_weights. "
                f"Valid types: {sorted(self._source_type_weights.keys())}"
            )
        return self._source_type_weights[source_type]

    async def _check_duplicate(self, hash_value: str, knowledge_id: str | None) -> None:
        if not self._metadata_store:
            return
        existing = await self._metadata_store.find_by_hash(hash_value, knowledge_id)
        if existing is not None:
            raise DuplicateSourceError(
                f"File already ingested as source {existing.source_id} (hash={hash_value[:12]}...)"
            )

    async def _dispatch_methods(
        self,
        source_id: str,
        knowledge_id: str | None,
        source_type: str | None,
        source_weight: float,
        title: str,
        full_text: str,
        chunks: list,
        tags: list[str],
        metadata: dict[str, Any],
        hash_value: str | None = None,
        pages: list[ParsedPage] | None = None,
        on_progress: IngestionProgress | None = None,
        notes: list[str] | None = None,
    ) -> None:
        """Dispatch ingestion to all registered methods in parallel groups.

        Each method declares ``required: bool``. Required methods (vector,
        document) run concurrently via ``asyncio.TaskGroup`` — a single
        failure cancels all siblings and aborts ingestion with
        ``IngestionError``, skipping the metadata commit so no partially-
        ingested source is ever marked as successful.

        Optional methods (graph, tree) run concurrently via
        ``asyncio.gather(return_exceptions=True)`` — failures are logged and
        the pipeline continues.

        A method missing the ``required`` attribute is treated as required
        to preserve data integrity by default.

        ``on_progress`` fires once per non-empty group with a boundary
        progress event, plus a final ``(total, total)`` event on successful
        completion — regardless of which groups were configured.
        """
        required = [m for m in self._ingestion_methods if getattr(m, "required", True)]
        optional = [m for m in self._ingestion_methods if not getattr(m, "required", True)]
        total = len(required) + len(optional)

        ingest_kwargs: dict[str, Any] = dict(
            source_id=source_id,
            knowledge_id=knowledge_id,
            source_type=source_type,
            source_weight=source_weight,
            title=title,
            full_text=full_text,
            chunks=chunks,
            tags=tags,
            metadata=metadata,
            hash_value=hash_value,
            pages=pages,
            notes=notes,
        )

        if required:
            try:
                async with asyncio.TaskGroup() as tg:
                    for m in required:
                        tg.create_task(m.ingest(**ingest_kwargs), name=m.name)
            except* (KeyboardInterrupt, SystemExit, asyncio.CancelledError) as eg:
                # Non-failure BaseExceptions must propagate unwrapped so shutdown/
                # interruption still cancels the surrounding task cleanly.
                raise eg.exceptions[0] from None
            except* Exception as eg:
                for exc in eg.exceptions:
                    logger.error("required ingestion method failed — aborting: %s", exc, exc_info=exc)
                causes = "; ".join(str(e) for e in eg.exceptions)
                raise IngestionError(f"required ingestion method failed: {causes}") from eg.exceptions[0]

        # Fire mid-progress after the required group if it did any work.
        if required and on_progress is not None:
            await on_progress(len(required), total)

        if optional:
            outcomes = await asyncio.gather(
                *(m.ingest(**ingest_kwargs) for m in optional),
                return_exceptions=True,
            )
            for method, outcome in zip(optional, outcomes, strict=True):
                if isinstance(outcome, BaseException):
                    logger.warning("optional ingestion method '%s' failed: %s", method.name, outcome)

        # Unconditional tail event so callers watching for `done == total`
        # see completion regardless of which groups were configured.
        if on_progress is not None:
            await on_progress(total, total)

    async def ingest(
        self,
        file_path: str | Path,
        knowledge_id: str | None = None,
        source_type: str | None = None,
        metadata: dict[str, Any] | None = None,
        page_range: str | None = None,
        resume_from_chunk: int = 0,
        on_progress: IngestionProgress | None = None,
    ) -> Source:
        """Ingest a file with auto-detection: PDF, text, markdown, or image."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if not file_path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")

        ext = file_path.suffix.lower()
        source_weight = self._resolve_weight(source_type)
        metadata = metadata or {}

        hash_value = await asyncio.to_thread(file_hash, file_path)
        if resume_from_chunk == 0:
            await self._check_duplicate(hash_value, knowledge_id)

        pages_filter = parse_page_range(page_range) if page_range else None

        if ext in IMAGE_EXTENSIONS:
            if not self._vision_parser:
                raise ConfigurationError("vision provider required for image ingestion")
            logger.info("processing file: %s (%s, image)", file_path.name, ext)
            pages = await self._vision_parser.parse(str(file_path), pages=pages_filter)
        elif ext in FILE_PARSERS_BY_EXTENSION:
            logger.info("processing file: %s (%s)", file_path.name, ext)
            parser_cls = FILE_PARSERS_BY_EXTENSION[ext]
            parser = parser_cls()
            pages = parser.parse(str(file_path), pages=pages_filter)
        else:
            raise ValueError(f"Unsupported file type: {ext}. Supported: {sorted(FILE_PARSERS_BY_EXTENSION.keys())}")

        if not pages:
            raise EmptyDocumentError(
                f"Document produced no content to ingest: {file_path.name}",
                reason="no_text_content",
            )

        source_id = str(uuid4())

        chunks = self._chunker.chunk(pages)
        logger.info("%d chunks from %d pages", len(chunks), len(pages))

        if not chunks:
            raise EmptyDocumentError(f"Document produced no content to ingest: {file_path.name}")

        if self._chunk_context_headers:
            source_name = (metadata or {}).get("name", file_path.name)
            chunks = contextualize_chunks(chunks, source_name=source_name, source_type=source_type)

        full_text = "\n\n".join(f"[Page {p.page_number}]\n{p.content}" for p in pages)

        notes: list[str] = []

        if self._contextual_chunk is not None and self._contextual_chunk.enabled:
            try:
                chunks = await contextualize_chunks_with_llm(
                    chunks,
                    document_text=full_text,
                    config=self._contextual_chunk,
                )
            except EnrichmentSkipped as exc:
                logger.warning("ingestion enrichment skipped: %s", exc)
                await record_skip(notes, step=exc.step, level="info", reason=exc.reason)
                if exc.step == "contextual_chunk":
                    set_ingest_field("contextual_chunk_skipped", True)

        if self._document_expansion is not None and self._document_expansion.enabled:
            assert self._expansion_registry is not None  # guaranteed by KnowledgeEngine.__init__
            chunks = await expand_chunks(
                chunks,
                self._document_expansion,
                self._expansion_registry,
                notes=notes,
            )

        title = metadata.get("name", file_path.name)

        # Stamp the per-source token estimate into the metadata blob so the
        # routing layer can sum corpus size without re-reading content. Sum of
        # per-page counts (not count_tokens(full_text)) so the page-marker
        # decoration we added above doesn't inflate the number.
        metadata["estimated_tokens"] = (
            sum(self._token_counter.count(p.content) for p in pages) if self._token_counter is not None else 0
        )

        await self._dispatch_methods(
            source_id=source_id,
            knowledge_id=knowledge_id,
            source_type=source_type,
            source_weight=source_weight,
            title=title,
            full_text=full_text,
            chunks=chunks,
            tags=[],
            metadata=metadata,
            hash_value=hash_value,
            pages=pages,
            on_progress=on_progress,
            notes=notes,
        )

        if notes:
            metadata.setdefault("ingestion_notes", []).extend(notes)

        source = Source(
            source_id=source_id,
            metadata=metadata,
            tags=[],
            chunk_count=len(chunks),
            embedding_model=self._embedding_model_name,
            file_hash=hash_value,
            created_at=datetime.now(UTC),
            knowledge_id=knowledge_id,
            source_type=source_type,
            source_weight=source_weight,
        )

        if self._metadata_store:
            await self._metadata_store.create_source(source)

        if self._on_ingestion_complete:
            await self._on_ingestion_complete(knowledge_id)

        logger.info("complete: %s --> %d chunks, source_id=%s", file_path.name, len(chunks), source_id)
        return source

    async def ingest_text(
        self,
        content: str,
        knowledge_id: str | None = None,
        source_type: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Source:
        source_weight = self._resolve_weight(source_type)
        metadata = metadata or {}

        logger.info("text ingestion started: %d chars", len(content))

        pages = [ParsedPage(page_number=1, content=content)]
        chunks = self._chunker.chunk(pages)

        if not chunks:
            raise EmptyDocumentError("Text content produced no chunks to ingest")

        if self._chunk_context_headers:
            source_name = (metadata or {}).get("name", "text-input")
            chunks = contextualize_chunks(chunks, source_name=source_name, source_type=source_type)

        notes: list[str] = []

        if self._contextual_chunk is not None and self._contextual_chunk.enabled:
            try:
                chunks = await contextualize_chunks_with_llm(
                    chunks,
                    document_text=content,
                    config=self._contextual_chunk,
                )
            except EnrichmentSkipped as exc:
                logger.warning("ingestion enrichment skipped: %s", exc)
                await record_skip(notes, step=exc.step, level="info", reason=exc.reason)
                if exc.step == "contextual_chunk":
                    set_ingest_field("contextual_chunk_skipped", True)

        if self._document_expansion is not None and self._document_expansion.enabled:
            assert self._expansion_registry is not None  # guaranteed by KnowledgeEngine.__init__
            chunks = await expand_chunks(
                chunks,
                self._document_expansion,
                self._expansion_registry,
                notes=notes,
            )

        source_id = str(uuid4())
        title = metadata.get("name", "text-input")

        metadata["estimated_tokens"] = self._token_counter.count(content) if self._token_counter is not None else 0

        await self._dispatch_methods(
            source_id=source_id,
            knowledge_id=knowledge_id,
            source_type=source_type,
            source_weight=source_weight,
            title=title,
            full_text=content,
            chunks=chunks,
            tags=[],
            metadata=metadata,
            hash_value=None,
            pages=pages,
            notes=notes,
        )

        if notes:
            metadata.setdefault("ingestion_notes", []).extend(notes)

        source = Source(
            source_id=source_id,
            metadata=metadata,
            tags=[],
            chunk_count=len(chunks),
            embedding_model=self._embedding_model_name,
            file_hash=None,
            created_at=datetime.now(UTC),
            knowledge_id=knowledge_id,
            source_type=source_type,
            source_weight=source_weight,
        )

        if self._metadata_store:
            await self._metadata_store.create_source(source)

        if self._on_ingestion_complete:
            await self._on_ingestion_complete(knowledge_id)

        logger.info("text ingestion complete: source=%s, chunks=%d", source_id, len(chunks))
        return source
