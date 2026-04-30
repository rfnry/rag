from rfnry_rag.common.errors import ConfigurationError as ConfigurationError
from rfnry_rag.common.errors import SdkBaseError


class RagError(SdkBaseError):
    """Base exception for retrieval SDK errors."""


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


class RetrievalError(RagError):
    """Error during retrieval pipeline."""


class GenerationError(RagError):
    """Error during LLM generation."""


class StoreError(RagError):
    """Error from a storage backend."""


class DuplicateSourceError(StoreError):
    """Attempted to create a source with an existing ID."""


class SourceNotFoundError(StoreError):
    """Source ID does not exist."""


class InputError(RagError, ValueError):
    """Raised when a public-input guard rejects caller-supplied text or metadata.

    Inherits from both RagError (for catching SDK-specific errors) and
    ValueError (for back-compat — existing `except ValueError:` still works).
    """
