"""DrawingIngestionConfig: pluggable knobs for the DrawingIngestion pipeline.

Every domain-specific choice (symbol vocabularies, off-page-connector regex
patterns, relation-type vocabulary, fuzzy-matching thresholds) is consumer-
configurable. The SDK ships sensible defaults (IEC 60617 electrical + ISA 5.1
P&ID) but every knob is overridable.

Passed as the ``config=`` argument to ``DrawingIngestion``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from rfnry_knowledge.exceptions import ConfigurationError
from rfnry_knowledge.ingestion.drawing.defaults import (
    DEFAULT_OFF_PAGE_CONNECTOR_PATTERNS,
    DEFAULT_RELATION_VOCABULARY,
    DEFAULT_SYMBOL_LIBRARY,
)
from rfnry_knowledge.providers import LLMClient


@dataclass
class DrawingIngestionConfig:
    """Configuration for the DrawingIngestion pipeline.

    Disabled by default. When ``enabled=False`` the config is inert — bounds
    checks are skipped so pathological values do not raise. This mirrors the
    tree-search configs.

    When ``enabled=True``, ``__post_init__`` materializes the resolved
    ``symbol_library`` / ``off_page_connector_patterns`` / ``relation_vocabulary``
    onto the instance so callers can read them directly (no ``resolved_*``
    property dance). The consumer-facing API:

    - Pass ``symbol_library={...}`` to FULLY REPLACE the defaults.
    - Pass ``symbol_library_extensions={...}`` to ADD to the defaults.
    - Leave both ``None`` to use IEC 60617 + ISA 5.1 defaults as-is.
    """

    enabled: bool = False
    lm_client: LLMClient | None = None
    dpi: int = 400
    page_image_format: Literal["png", "jpeg"] = "png"
    default_domain: Literal["auto", "electrical", "p_and_id", "mechanical", "mixed"] = "auto"
    symbol_library: dict[str, list[str]] | None = None
    symbol_library_extensions: dict[str, list[str]] | None = None
    analyze_concurrency: int = 5
    off_page_connector_patterns: list[str] | None = None
    sheet_set_grouping: Literal["all", "none", "by_title_block", "explicit"] = "all"
    # unbounded: list of page-number lists sized by consumer's PDF; not a scalar threshold.
    explicit_sheet_groups: list[list[int]] = field(default_factory=list)  # unbounded: see above
    relation_vocabulary: dict[str, str] | None = None
    graph_write_batch_size: int = 500

    def __post_init__(self) -> None:
        if not self.enabled:
            # Inert: skip bounds + materialization so disabled configs with
            # pathological values (e.g. dpi=9999) don't raise.
            return

        # --- Enum-ish literal validation (dataclass can't enforce at runtime) ---
        if self.page_image_format not in ("png", "jpeg"):
            raise ConfigurationError(
                f"DrawingIngestionConfig.page_image_format must be 'png' or 'jpeg', got {self.page_image_format!r}"
            )
        if self.default_domain not in ("auto", "electrical", "p_and_id", "mechanical", "mixed"):
            raise ConfigurationError(
                f"DrawingIngestionConfig.default_domain must be one of "
                f"'auto'/'electrical'/'p_and_id'/'mechanical'/'mixed', got {self.default_domain!r}"
            )
        if self.sheet_set_grouping not in ("all", "none", "by_title_block", "explicit"):
            raise ConfigurationError(
                f"DrawingIngestionConfig.sheet_set_grouping must be one of "
                f"'all'/'none'/'by_title_block'/'explicit', got {self.sheet_set_grouping!r}"
            )

        # --- Numeric bounds ---
        # dpi: lower bound 150 keeps symbols legible for vision LLMs; upper bound
        # 600 caps per-page buffer growth (each page can exceed 100MB beyond that).
        if not (150 <= self.dpi <= 600):
            raise ConfigurationError(f"DrawingIngestionConfig.dpi={self.dpi} out of range [150, 600]")
        if not (1 <= self.analyze_concurrency <= 100):
            raise ConfigurationError(
                f"DrawingIngestionConfig.analyze_concurrency={self.analyze_concurrency} out of range [1, 100]"
            )
        if not (1 <= self.graph_write_batch_size <= 10_000):
            raise ConfigurationError(
                f"DrawingIngestionConfig.graph_write_batch_size={self.graph_write_batch_size} out of range [1, 10_000]"
            )

        # --- Materialize resolved views onto the fields ---
        # symbol_library: full replace (consumer passed dict) OR defaults + extensions.
        if self.symbol_library is None:
            merged: dict[str, list[str]] = {k: list(v) for k, v in DEFAULT_SYMBOL_LIBRARY.items()}
            for domain, extras in (self.symbol_library_extensions or {}).items():
                merged.setdefault(domain, []).extend(extras)
            self.symbol_library = merged
        # When symbol_library IS provided, extensions are ignored (full-replace semantics).

        if self.off_page_connector_patterns is None:
            self.off_page_connector_patterns = list(DEFAULT_OFF_PAGE_CONNECTOR_PATTERNS)

        if self.relation_vocabulary is None:
            self.relation_vocabulary = dict(DEFAULT_RELATION_VOCABULARY)

        # --- Validate relation_vocabulary targets against Neo4j allowlist ---
        # Late import: stores.graph.neo4j pulls in heavy deps we don't want at
        # dataclass-decoration time.
        from rfnry_knowledge.stores.graph.neo4j import ALLOWED_RELATION_TYPES

        for wire_style, rel in self.relation_vocabulary.items():
            if rel not in ALLOWED_RELATION_TYPES:
                raise ConfigurationError(
                    f"DrawingIngestionConfig.relation_vocabulary[{wire_style!r}]={rel!r} "
                    f"is not in ALLOWED_RELATION_TYPES ({sorted(ALLOWED_RELATION_TYPES)})"
                )
