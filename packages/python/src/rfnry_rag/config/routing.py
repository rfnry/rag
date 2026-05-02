from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from rfnry_rag.exceptions import ConfigurationError


class QueryMode(Enum):
    """Per-query routing strategy.

    `INDEXED` runs the standard indexed pipeline. `FULL_CONTEXT` skips
    retrieval and loads the full corpus into a prompt-cached prefix. `AUTO`
    dispatches between the two based on `full_context_threshold` versus
    corpus tokens.
    """

    INDEXED = "indexed"
    FULL_CONTEXT = "full_context"
    AUTO = "auto"


@dataclass
class RoutingConfig:
    """Top-level routing strategy. AUTO is the recommended mode: as context
    windows grow, more corpora cross the threshold and shift to FULL_CONTEXT
    transparently.

    Default ``full_context_threshold=150_000`` is Anthropic's "stuff the whole
    knowledge base in the prompt" recommendation of ~200k tokens minus
    ~25% headroom for the system prompt, chat history, the user question, and
    the answer. Do not raise it to match a model's advertised window: Liu et al.
    2023 ("Lost in the Middle") and Li et al. 2025 ("LaRA") both show that
    effective context is meaningfully lower than advertised, with U-shaped
    accuracy that worsens for weaker models. When a generation
    ``GenerativeModel.context_size`` is declared, RagEngine init asserts
    that the threshold + reserve fits — see
    ``RagEngine._validate_full_context_fits_provider_window``.
    """

    mode: QueryMode = QueryMode.INDEXED
    full_context_threshold: int = 150_000

    def __post_init__(self) -> None:
        if not (1_000 <= self.full_context_threshold <= 2_000_000):
            raise ConfigurationError(
                f"RoutingConfig.full_context_threshold={self.full_context_threshold} out of range [1_000, 2_000_000]"
            )
