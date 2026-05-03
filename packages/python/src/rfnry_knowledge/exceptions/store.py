from rfnry_knowledge.exceptions.base import KnowledgeEngineError


class StoreError(KnowledgeEngineError):
    """Error from a storage backend."""


class DuplicateSourceError(StoreError):
    """Attempted to create a source with an existing ID."""


class SourceNotFoundError(StoreError):
    """Source ID does not exist."""
