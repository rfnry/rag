"""Contract: every numeric config field must be either (a) referenced by name in
its class's __post_init__ (indicating validation) or (b) have a
``# unbounded: <reason>`` comment on the field line justifying its absence.
Fails on any new numeric field that has neither.
"""

import inspect
import re
from dataclasses import fields
from typing import Any

from rfnry_rag.config.drawing import DrawingIngestionConfig
from rfnry_rag.config.graph import GraphIngestionConfig
from rfnry_rag.ingestion.chunk.batch import BatchConfig
from rfnry_rag.observability.benchmark import BenchmarkConfig
from rfnry_rag.providers import LanguageModelClient
from rfnry_rag.retrieval.search.rewriting.multi_query import MultiQueryRewriting
from rfnry_rag.server import (
    DocumentExpansionConfig,
    GenerationConfig,
    IngestionConfig,
    RetrievalConfig,
    RoutingConfig,
)

_CONFIGS_TO_AUDIT: list[type] = [
    IngestionConfig,
    RetrievalConfig,
    GenerationConfig,
    DrawingIngestionConfig,
    GraphIngestionConfig,
    LanguageModelClient,
    BatchConfig,
    MultiQueryRewriting,
    DocumentExpansionConfig,
    BenchmarkConfig,
    RoutingConfig,
]


def _is_numeric(field_type: Any) -> bool:
    """Check if the field type is numeric (int, float, or Optional of those)."""
    s = str(field_type)
    return any(t in s for t in ("int", "float"))


def test_every_numeric_config_field_has_bounds_or_marker() -> None:
    violations: list[str] = []
    for cls in _CONFIGS_TO_AUDIT:
        try:
            src = inspect.getsource(cls)
        except OSError:
            continue
        post_init = inspect.getsource(cls.__post_init__) if hasattr(cls, "__post_init__") else ""

        for f in fields(cls):
            if not _is_numeric(f.type):
                continue
            # Either the field name appears in __post_init__ (validation) or the
            # class source has a "# unbounded: <reason>" comment on the line with
            # the field declaration.
            has_validation = bool(re.search(rf"\bself\.{re.escape(f.name)}\b", post_init))
            has_unbounded_marker = bool(
                re.search(
                    rf"^\s*{re.escape(f.name)}\s*:.*#\s*unbounded:",
                    src,
                    re.MULTILINE,
                )
            )
            if not (has_validation or has_unbounded_marker):
                violations.append(
                    f"{cls.__module__}.{cls.__name__}.{f.name} — no bounds validation or '# unbounded: <reason>' marker"
                )

    assert not violations, "config-bounds violations:\n  " + "\n  ".join(violations)
