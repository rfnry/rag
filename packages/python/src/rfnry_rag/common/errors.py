"""Shared error base classes for both retrieval and reasoning SDKs."""


class SdkBaseError(Exception):
    """Base class for all rfnry-rag SDK errors.

    Not re-exported at the top-level of `rfnry_rag` to avoid any ambiguity
    with the Python builtin `BaseException`. Users should catch the
    specific subclasses (`RagError`, `ReasoningError`, `ConfigurationError`)
    or import this from `rfnry_rag.common.errors` explicitly.
    """


class ConfigurationError(SdkBaseError):
    """Invalid SDK configuration."""
