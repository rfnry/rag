"""Contract: every numeric config field must be either (a) referenced by name in
its class's __post_init__ (indicating validation) or (b) have a
``# unbounded: <reason>`` comment on the field line justifying its absence.
Fails on any new numeric field that has neither.

Plain (non-dataclass) classes that own bounds are also audited via their
``__init__`` parameter list — the same rule applies, with validation
detected from the ``__init__`` body and the unbounded marker placed on
the parameter line.
"""

import inspect
import re
from dataclasses import fields
from typing import Any

from rfnry_rag.config import (
    DocumentExpansionConfig,
    GenerationConfig,
    IngestionConfig,
    RetrievalConfig,
    RoutingConfig,
)
from rfnry_rag.config.drawing import DrawingIngestionConfig
from rfnry_rag.config.graph import GraphIngestionConfig
from rfnry_rag.ingestion.chunk.batch import BatchConfig
from rfnry_rag.ingestion.methods.analyzed import AnalyzedIngestion
from rfnry_rag.observability.benchmark import BenchmarkConfig
from rfnry_rag.providers import LanguageModel, LanguageModelClient
from rfnry_rag.retrieval.methods.vector import VectorRetrieval

_CONFIGS_TO_AUDIT: list[type] = [
    IngestionConfig,
    RetrievalConfig,
    GenerationConfig,
    DrawingIngestionConfig,
    GraphIngestionConfig,
    LanguageModelClient,
    LanguageModel,
    BatchConfig,
    DocumentExpansionConfig,
    BenchmarkConfig,
    RoutingConfig,
]

# Plain classes (non-dataclass) that own bounds on their ``__init__``
# parameters. We audit these explicitly: dataclass introspection via
# ``dataclasses.fields()`` does not reach them.
_INIT_PARAM_BOUNDED_CLASSES: list[type] = [
    AnalyzedIngestion,
    VectorRetrieval,
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


_CONTAINER_TYPES = ("dict", "list", "tuple", "set", "Sequence", "Mapping")


def _is_scalar_numeric(annotation: str) -> bool:
    """``int`` / ``float`` (optionally ``| None``) but not nested in a container."""
    if any(t in annotation for t in _CONTAINER_TYPES):
        return False
    return _is_numeric(annotation)


def _numeric_init_params(cls: type) -> list[str]:
    """Names of ``__init__`` parameters whose annotation is a scalar numeric."""
    sig = inspect.signature(cls.__init__)
    out: list[str] = []
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        ann = str(param.annotation)
        if _is_scalar_numeric(ann):
            out.append(name)
    return out


def _audit_init_param_bounds(cls: type) -> list[str]:
    """Return one violation string per unmarked, unvalidated numeric ``__init__``
    parameter on ``cls``. Empty list means the class is clean."""
    try:
        src = inspect.getsource(cls)
    except OSError:
        return []
    init_body = inspect.getsource(cls.__init__)
    violations: list[str] = []
    for name in _numeric_init_params(cls):
        # Validation hint: the parameter name appears inside a comparison in
        # the __init__ body (e.g. ``if not (1 <= x <= 100)`` or ``x > 0``).
        has_validation = bool(re.search(rf"<=\s*{re.escape(name)}\s*<=", init_body)) or bool(
            re.search(rf"\b{re.escape(name)}\b\s*[<>!]=?", init_body)
        )
        has_unbounded_marker = bool(
            re.search(
                rf"^\s*{re.escape(name)}\s*:.*#\s*unbounded:",
                src,
                re.MULTILINE,
            )
        )
        if not (has_validation or has_unbounded_marker):
            violations.append(
                f"{cls.__module__}.{cls.__name__}.{name} — no bounds validation or '# unbounded: <reason>' marker"
            )
    return violations


def test_every_numeric_init_param_has_bounds_or_marker() -> None:
    """Same rule as the dataclass audit, but for plain classes whose bounded
    state lives on ``__init__`` parameters."""
    violations: list[str] = []
    for cls in _INIT_PARAM_BOUNDED_CLASSES:
        violations.extend(_audit_init_param_bounds(cls))

    assert not violations, "init-bounds violations:\n  " + "\n  ".join(violations)


def test_init_bounds_contract_catches_unbounded_addition() -> None:
    """Sanity: the audit detects an unmarked numeric ``__init__`` param on a
    synthetic plain class. If the audit machinery (``_numeric_init_params`` /
    ``_audit_init_param_bounds``) drifts and stops flagging real violations,
    this self-test catches it."""

    class _Probe:
        def __init__(self, x: int = 5) -> None:
            self._x = x

    violations = _audit_init_param_bounds(_Probe)
    assert any("_Probe" in v and ".x " in f"{v} " for v in violations), (
        f"audit failed to flag _Probe.x as unbounded; got: {violations!r}"
    )
