from uuid import uuid4

from rfnry_rag.retrieval.modules.ingestion.chunk.splitter import RecursiveTextSplitter
from rfnry_rag.retrieval.modules.ingestion.models import ChunkedContent, ParsedPage


class SemanticChunker:
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        parent_chunk_size: int = 0,
        parent_chunk_overlap: int = 200,
    ) -> None:
        self._child_splitter = RecursiveTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self._parent_splitter = None
        if parent_chunk_size > 0:
            self._parent_splitter = RecursiveTextSplitter(
                chunk_size=parent_chunk_size,
                chunk_overlap=parent_chunk_overlap,
            )

    def chunk(self, pages: list[ParsedPage]) -> list[ChunkedContent]:
        if self._parent_splitter:
            return self._chunk_parent_child(pages)
        return self._chunk_flat(pages)

    def _chunk_flat(self, pages: list[ParsedPage]) -> list[ChunkedContent]:
        chunks: list[ChunkedContent] = []
        global_index = 0
        for page in pages:
            texts = self._child_splitter.split_text(page.content)
            for text in texts:
                chunks.append(
                    ChunkedContent(
                        content=text,
                        page_number=page.page_number,
                        section=None,
                        chunk_index=global_index,
                    )
                )
                global_index += 1
        return chunks

    def _chunk_parent_child(self, pages: list[ParsedPage]) -> list[ChunkedContent]:
        chunks: list[ChunkedContent] = []
        global_index = 0
        assert self._parent_splitter is not None, "_chunk_parent_child called without parent_splitter"

        for page in pages:
            parent_texts = self._parent_splitter.split_text(page.content)

            for parent_text in parent_texts:
                parent_id = str(uuid4())

                chunks.append(
                    ChunkedContent(
                        content=parent_text,
                        page_number=page.page_number,
                        section=None,
                        chunk_index=global_index,
                        chunk_type="parent",
                        parent_id=parent_id,
                    )
                )
                global_index += 1

                child_texts = self._child_splitter.split_text(parent_text)
                for child_text in child_texts:
                    chunks.append(
                        ChunkedContent(
                            content=child_text,
                            page_number=page.page_number,
                            section=None,
                            chunk_index=global_index,
                            chunk_type="child",
                            parent_id=parent_id,
                        )
                    )
                    global_index += 1

        return chunks
