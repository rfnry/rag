from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from rfnry_rag.retrieval.common.errors import (
    ConfigurationError,
    DuplicateSourceError,
    EmptyDocumentError,
    IngestionError,
)
from rfnry_rag.retrieval.common.hashing import file_hash
from rfnry_rag.retrieval.common.logging import get_logger
from rfnry_rag.retrieval.common.models import Source
from rfnry_rag.retrieval.common.page_range import parse_page_range
from rfnry_rag.retrieval.modules.ingestion.chunk.chunker import SemanticChunker
from rfnry_rag.retrieval.modules.ingestion.chunk.context import contextualize_chunks
from rfnry_rag.retrieval.modules.ingestion.chunk.parsers.pdf import PDFParser
from rfnry_rag.retrieval.modules.ingestion.chunk.parsers.text import TextParser
from rfnry_rag.retrieval.modules.ingestion.models import ParsedPage
from rfnry_rag.retrieval.modules.ingestion.vision.base import BaseVision
from rfnry_rag.retrieval.modules.ingestion.vision.constants import IMAGE_EXTENSIONS
from rfnry_rag.retrieval.stores.metadata.base import BaseMetadataStore

if TYPE_CHECKING:
    from rfnry_rag.retrieval.modules.ingestion.base import BaseIngestionMethod

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
        contextual_chunking: bool = True,
    ) -> None:
        self._chunker = chunker
        self._ingestion_methods = ingestion_methods
        self._metadata_store = metadata_store
        self._embedding_model_name = embedding_model_name
        self._source_type_weights = source_type_weights
        self._on_ingestion_complete = on_ingestion_complete
        self._vision_parser = vision_parser
        self._contextual_chunking = contextual_chunking

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
        existing = await self._metadata_store.list_sources(knowledge_id=knowledge_id)
        for source in existing:
            if source.file_hash == hash_value:
                raise DuplicateSourceError(
                    f"File already ingested as source {source.source_id} (hash={hash_value[:12]}...)"
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
    ) -> None:
        """Dispatch ingestion to all registered methods.

        Each method declares ``required: bool``. Required methods (vector,
        document) abort ingestion on failure by raising ``IngestionError`` —
        the caller uses this to skip the metadata commit so a partially-
        ingested source is never marked as successful. Optional methods
        (graph, tree) are logged and the pipeline continues.

        A method missing the ``required`` attribute is treated as required
        to preserve data integrity by default.

        ``on_progress`` is invoked with ``(done, total)`` after each
        successful method call. Optional methods that fail are skipped and
        do NOT emit a progress event, so ``total`` may not be reached if
        every optional method raises.
        """
        total = len(self._ingestion_methods)
        for done, method in enumerate(self._ingestion_methods, start=1):
            try:
                await method.ingest(
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
                )
            except Exception as exc:
                if getattr(method, "required", True):
                    logger.exception("required ingestion method '%s' failed — aborting", method.name)
                    raise IngestionError(f"required ingestion method '{method.name}' failed: {exc}") from exc
                logger.warning("optional ingestion method '%s' failed: %s", method.name, exc)
                continue
            if on_progress is not None:
                await on_progress(done, total)

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

        if self._contextual_chunking:
            source_name = (metadata or {}).get("name", file_path.name)
            chunks = contextualize_chunks(chunks, source_name=source_name, source_type=source_type)

        full_text = "\n\n".join(f"[Page {p.page_number}]\n{p.content}" for p in pages)
        title = metadata.get("name", file_path.name)

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
        )

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

        if self._contextual_chunking:
            source_name = (metadata or {}).get("name", "text-input")
            chunks = contextualize_chunks(chunks, source_name=source_name, source_type=source_type)

        source_id = str(uuid4())
        title = metadata.get("name", "text-input")

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
        )

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
