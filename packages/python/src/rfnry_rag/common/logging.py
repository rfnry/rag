import logging
import os

_BAML_LOG_ENV = "BAML_LOG"
_RFNRY_RAG_BAML_LOG_ENV = "RFNRY_RAG_BAML_LOG"

_VALID_LEVELS = set(logging.getLevelNamesMapping())
_VALID_BAML_LEVELS = {"trace", "debug", "info", "warn", "error", "off"}


def _resolve_level(raw: str) -> int:
    """Resolve a level name string to its integer value.

    Raises ``ConfigurationError`` for unrecognised names so invalid
    ``RFNRY_RAG_LOG_LEVEL`` values are caught at startup rather than
    silently defaulting to NOTSET (0).
    """
    upper = raw.upper()
    if upper not in _VALID_LEVELS:
        from rfnry_rag.common.errors import ConfigurationError

        raise ConfigurationError(f"unknown log level {raw!r}; valid: {sorted(_VALID_LEVELS)}")
    return logging.getLevelNamesMapping()[upper]


def query_logging_enabled() -> bool:
    """True when user queries may be logged verbatim. Defaults False (PII-safe).

    Callers that log raw query text must gate behind this so no module leaks
    user inputs at INFO when the env var is unset."""
    return os.environ.get("RFNRY_RAG_LOG_QUERIES", "").lower() == "true"


def _propagate_baml_log_env() -> None:
    """Wire RFNRY_RAG_BAML_LOG to BAML_LOG (what BAML actually reads) so users have
    one namespaced env var instead of touching BAML's internal name directly.

    An explicitly-set BAML_LOG wins — we only propagate when BAML_LOG is unset.
    """
    raw = os.environ.get(_RFNRY_RAG_BAML_LOG_ENV)
    if raw is not None:
        if raw.lower() not in _VALID_BAML_LEVELS:
            from rfnry_rag.common.errors import ConfigurationError

            raise ConfigurationError(
                f"RFNRY_RAG_BAML_LOG must be one of {sorted(_VALID_BAML_LEVELS)}, got {raw!r}"
            )
        if not os.getenv(_BAML_LOG_ENV):
            os.environ[_BAML_LOG_ENV] = raw.lower()


def get_logger(module: str) -> logging.Logger:
    _propagate_baml_log_env()
    logger = logging.getLogger(f"rfnry_rag.{module}")
    if os.getenv("RFNRY_RAG_LOG_ENABLED", "false").lower() == "true":
        level = os.getenv("RFNRY_RAG_LOG_LEVEL", "INFO")
        logger.setLevel(_resolve_level(level))
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
            logger.addHandler(handler)
            logger.propagate = False
    else:
        logger.setLevel(logging.CRITICAL)
    return logger
