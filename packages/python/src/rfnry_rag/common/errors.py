"""Shared error base classes for both retrieval and reasoning SDKs."""


class BaseException(Exception):
    """Base exception for all rfnry-rag SDK errors."""


class ConfigurationError(BaseException):
    """Invalid SDK configuration."""
