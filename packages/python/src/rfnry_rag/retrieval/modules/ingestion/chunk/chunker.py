from typing import Literal
from uuid import uuid4

from rfnry_rag.retrieval.modules.ingestion.chunk.splitter import RecursiveTextSplitter
from rfnry_rag.retrieval.modules.ingestion.chunk.structure import (
    build_heading_spans,
    find_atomic_regions,
    section_path_at,
)
from rfnry_rag.retrieval.modules.ingestion.chunk.token_counter import count_tokens
from rfnry_rag.retrieval.modules.ingestion.models import ChunkedContent, ParsedPage

ChunkSizeUnit = Literal["chars", "tokens"]


class SemanticChunker:
    def __init__(
        self,
        chunk_size: int = 375,
        chunk_overlap: int = 40,
        parent_chunk_size: int = 0,
        parent_chunk_overlap: int = 150,
        chunk_size_unit: ChunkSizeUnit = "tokens",
    ) -> None:
        if chunk_size_unit not in ("chars", "tokens"):
            raise ValueError(f"chunk_size_unit must be 'chars' or 'tokens', got {chunk_size_unit!r}")
        self.chunk_size_unit = chunk_size_unit
        length_function = count_tokens if chunk_size_unit == "tokens" else len

        self._child_splitter = RecursiveTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
        )
        self._parent_splitter = None
        if parent_chunk_size > 0:
            self._parent_splitter = RecursiveTextSplitter(
                chunk_size=parent_chunk_size,
                chunk_overlap=parent_chunk_overlap,
                length_function=length_function,
            )

    def chunk(self, pages: list[ParsedPage]) -> list[ChunkedContent]:
        if self._parent_splitter:
            return self._chunk_parent_child(pages)
        return self._chunk_flat(pages)

    def _emit_free_text_chunks(
        self,
        free_text: str,
        page_offset: int,
        page: ParsedPage,
        heading_spans: list,
        chunks: list[ChunkedContent],
        global_index: int,
    ) -> int:
        """Split a free-text segment and append chunks with per-chunk section lookup.

        ``page_offset`` is the start of ``free_text`` within ``page.content``.
        Each produced chunk's section is resolved by finding its position in
        the page so that heading transitions within a single free segment are
        respected.
        """
        search_from = page_offset
        for chunk_text, was_hard in self._child_splitter.split_text_with_flags(free_text):
            idx = page.content.find(chunk_text, search_from)
            chunk_offset = idx if idx != -1 else search_from
            search_from = chunk_offset + len(chunk_text)
            # Use the last character of the chunk for section lookup so that
            # the deepest heading within the chunk governs its section label.
            lookup_offset = chunk_offset + max(0, len(chunk_text) - 1)
            chunks.append(ChunkedContent(
                content=chunk_text,
                page_number=page.page_number,
                section=section_path_at(lookup_offset, heading_spans),
                chunk_index=global_index,
                was_hard_split=was_hard,
            ))
            global_index += 1
        return global_index

    def _chunk_flat(self, pages: list[ParsedPage]) -> list[ChunkedContent]:
        chunks: list[ChunkedContent] = []
        global_index = 0
        for page in pages:
            heading_spans = build_heading_spans(page.content)
            atomic_regions = find_atomic_regions(page.content)

            cursor = 0
            for region in atomic_regions:
                # Free text before this atomic region
                if cursor < region.start:
                    free_text = page.content[cursor:region.start]
                    global_index = self._emit_free_text_chunks(
                        free_text, cursor, page, heading_spans, chunks, global_index
                    )

                # Atomic region: emit as one chunk if it fits, else fall back to splitter
                if self._child_splitter._length_function(region.content) <= self._child_splitter._chunk_size:
                    chunks.append(ChunkedContent(
                        content=region.content,
                        page_number=page.page_number,
                        section=section_path_at(region.start, heading_spans),
                        chunk_index=global_index,
                        was_hard_split=False,
                    ))
                    global_index += 1
                else:
                    region_section = section_path_at(region.start, heading_spans)
                    for chunk_text, was_hard in self._child_splitter.split_text_with_flags(region.content):
                        chunks.append(ChunkedContent(
                            content=chunk_text,
                            page_number=page.page_number,
                            section=region_section,
                            chunk_index=global_index,
                            was_hard_split=was_hard,
                        ))
                        global_index += 1
                cursor = region.end

            # Trailing free text after the last atomic region
            if cursor < len(page.content):
                tail = page.content[cursor:]
                global_index = self._emit_free_text_chunks(
                    tail, cursor, page, heading_spans, chunks, global_index
                )
        return chunks

    def _chunk_parent_child(self, pages: list[ParsedPage]) -> list[ChunkedContent]:
        chunks: list[ChunkedContent] = []
        global_index = 0
        assert self._parent_splitter is not None, "_chunk_parent_child called without parent_splitter"

        for page in pages:
            heading_spans = build_heading_spans(page.content)
            parent_flagged = self._parent_splitter.split_text_with_flags(page.content)

            # Track the offset into page.content as we process each parent chunk.
            # Since parent chunks are produced by splitting page.content sequentially,
            # we find each parent's start offset by searching from a running cursor.
            parent_cursor = 0

            for parent_text, parent_hard_split in parent_flagged:
                parent_id = str(uuid4())

                # Locate this parent chunk's start offset in the page for section lookup.
                # search from parent_cursor to handle repeated substrings correctly.
                idx = page.content.find(parent_text, parent_cursor)
                parent_offset = idx if idx != -1 else parent_cursor
                parent_section = section_path_at(parent_offset, heading_spans)
                parent_cursor = parent_offset + len(parent_text)

                chunks.append(
                    ChunkedContent(
                        content=parent_text,
                        page_number=page.page_number,
                        section=parent_section,
                        chunk_index=global_index,
                        chunk_type="parent",
                        parent_id=parent_id,
                        was_hard_split=parent_hard_split,
                    )
                )
                global_index += 1

                for child_text, child_hard_split in self._child_splitter.split_text_with_flags(parent_text):
                    chunks.append(
                        ChunkedContent(
                            content=child_text,
                            page_number=page.page_number,
                            section=parent_section,
                            chunk_index=global_index,
                            chunk_type="child",
                            parent_id=parent_id,
                            was_hard_split=child_hard_split,
                        )
                    )
                    global_index += 1

        return chunks
