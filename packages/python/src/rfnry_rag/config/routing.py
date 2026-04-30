from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from rfnry_rag.exceptions import ConfigurationError


class QueryMode(Enum):
    """Per-query routing strategy.

    `RETRIEVAL` runs the standard indexed pipeline. `DIRECT` skips retrieval
    and loads the full corpus into a prompt-cached prefix. `AUTO` dispatches
    between the two based on `full_context_threshold` versus corpus tokens.
    """

    RETRIEVAL = "retrieval"
    DIRECT = "direct"
    AUTO = "auto"


@dataclass
class RoutingConfig:
    """Top-level routing strategy. AUTO is the recommended mode: as context
    windows grow, more corpora cross the threshold and shift to DIRECT
    transparently."""

    mode: QueryMode = QueryMode.RETRIEVAL
    full_context_threshold: int = 150_000

    def __post_init__(self) -> None:
        if not (1_000 <= self.full_context_threshold <= 2_000_000):
            raise ConfigurationError(
                f"RoutingConfig.full_context_threshold={self.full_context_threshold} out of range [1_000, 2_000_000]"
            )
