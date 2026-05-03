"""Exception hierarchy for rfnry-knowledge. Catch KnowledgeEngineError for the SDK catch-all."""

from rfnry_knowledge.exceptions.base import KnowledgeEngineError
from rfnry_knowledge.exceptions.configuration import ConfigurationError
from rfnry_knowledge.exceptions.generation import GenerationError
from rfnry_knowledge.exceptions.ingestion import (
    EmbeddingError,
    EmptyDocumentError,
    EnrichmentSkipped,
    IngestionError,
    IngestionInterruptedError,
    ParseError,
)
from rfnry_knowledge.exceptions.input import InputError
from rfnry_knowledge.exceptions.retrieval import RetrievalError
from rfnry_knowledge.exceptions.store import DuplicateSourceError, SourceNotFoundError, StoreError

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
    "KnowledgeEngineError",
    "ParseError",
    "RetrievalError",
    "SourceNotFoundError",
    "StoreError",
]
