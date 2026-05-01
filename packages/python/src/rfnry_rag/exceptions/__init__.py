"""Exception hierarchy for rfnry-rag. Catch RagError for the SDK catch-all."""

from rfnry_rag.exceptions.base import RagError
from rfnry_rag.exceptions.configuration import ConfigurationError
from rfnry_rag.exceptions.generation import GenerationError
from rfnry_rag.exceptions.ingestion import (
    EmbeddingError,
    EmptyDocumentError,
    EnrichmentSkipped,
    IngestionError,
    IngestionInterruptedError,
    ParseError,
)
from rfnry_rag.exceptions.input import InputError
from rfnry_rag.exceptions.retrieval import RetrievalError
from rfnry_rag.exceptions.store import DuplicateSourceError, SourceNotFoundError, StoreError

__all__ = [
    "ConfigurationError",
    "DuplicateSourceError",
    "EmbeddingError",
    "EmptyDocumentError",
    "EnrichmentSkipped",
    "GenerationError",
    "IngestionError",
    "IngestionInterruptedError",
    "InputError",
    "ParseError",
    "RagError",
    "RetrievalError",
    "SourceNotFoundError",
    "StoreError",
]
