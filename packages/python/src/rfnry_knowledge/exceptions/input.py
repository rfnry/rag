from rfnry_knowledge.exceptions.base import KnowledgeEngineError


class InputError(KnowledgeEngineError, ValueError):
    """Raised when a public-input guard rejects caller-supplied text or metadata.

    Inherits from both KnowledgeEngineError (catch-all SDK base) and ValueError so existing
    `except ValueError:` callers stay compatible.
    """
