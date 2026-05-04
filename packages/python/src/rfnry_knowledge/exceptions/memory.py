from rfnry_knowledge.exceptions.base import KnowledgeEngineError
from rfnry_knowledge.exceptions.ingestion import IngestionError
from rfnry_knowledge.exceptions.store import StoreError


class MemoryEngineError(KnowledgeEngineError):
    """Base for MemoryEngine errors."""


class MemoryNotFoundError(MemoryEngineError, StoreError):
    """update() / delete() targeted a memory_row_id that does not exist."""

    def __init__(self, message: str = "", *, memory_row_id: str = "") -> None:
        super().__init__(message)
        self.memory_row_id = memory_row_id


class MemoryExtractionError(MemoryEngineError, IngestionError):
    """The extractor failed to produce a usable memory list."""
