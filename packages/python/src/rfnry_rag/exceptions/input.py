from rfnry_rag.exceptions.base import RagError


class InputError(RagError, ValueError):
    """Raised when a public-input guard rejects caller-supplied text or metadata.

    Inherits from both RagError (catch-all SDK base) and ValueError so existing
    `except ValueError:` callers stay compatible.
    """
