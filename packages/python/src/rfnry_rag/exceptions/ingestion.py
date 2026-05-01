from rfnry_rag.exceptions.base import RagError


class IngestionError(RagError):
    """Error during document ingestion."""


class ParseError(IngestionError):
    """Error parsing a document (corrupt PDF, encoding issue, etc.)."""


class EmptyDocumentError(IngestionError):
    """Document produced no content to ingest."""

    def __init__(self, message: str = "", *, reason: str | None = None) -> None:
        super().__init__(message)
        self.reason = reason


class EmbeddingError(IngestionError):
    """Error generating embeddings."""


class IngestionInterruptedError(IngestionError):
    """Ingestion was interrupted mid-processing (e.g. rate limit, network error).

    Contains state needed to resume from the point of failure.
    """

    def __init__(
        self,
        message: str = "",
        *,
        completed_chunk_index: int = 0,
        source_id: str = "",
    ) -> None:
        super().__init__(message)
        self.completed_chunk_index = completed_chunk_index
        self.source_id = source_id


class EnrichmentSkipped(RagError):
    """Non-fatal pipeline step that produced no output and should be recorded as an audit note."""

    def __init__(self, step: str, reason: str) -> None:
        super().__init__(f"{step} skipped: {reason}")
        self.step = step
        self.reason = reason
