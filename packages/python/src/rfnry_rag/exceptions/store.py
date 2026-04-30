from rfnry_rag.exceptions.base import RagError


class StoreError(RagError):
    """Error from a storage backend."""


class DuplicateSourceError(StoreError):
    """Attempted to create a source with an existing ID."""


class SourceNotFoundError(StoreError):
    """Source ID does not exist."""
