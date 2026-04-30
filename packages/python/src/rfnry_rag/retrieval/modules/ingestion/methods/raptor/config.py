"""RaptorConfig — RAPTOR-style hierarchical summarization retrieval configuration.

Default-off configuration dataclass for RAPTOR. When ``enabled=False``
(the default) the rest of this module is byte-for-byte unused.
"""

from __future__ import annotations

from dataclasses import dataclass

from rfnry_rag.retrieval.common.errors import ConfigurationError
from rfnry_rag.retrieval.common.language_model import LanguageModelClient

# Allowlist for ``cluster_algorithm``. Adding a third clustering strategy
# requires explicit redesign — kmeans + hdbscan cover the partition-based and
# density-based regimes RAPTOR's research surveyed.
_VALID_CLUSTER_ALGORITHMS = frozenset({"kmeans", "hdbscan"})


@dataclass
class RaptorConfig:
    """RAPTOR-style hierarchical summarization retrieval configuration.

    Default-off. When ``enabled=True``, ``RagEngine.build_raptor_index(knowledge_id)``
    becomes available; building a tree clusters chunks under that
    knowledge_id, summarises each cluster via the configured ``summary_model``,
    recurses up to ``max_levels``, and persists summary vectors with
    ``vector_role="raptor_summary"``.

    Per-knowledge_id; blue/green atomic swap; consumer controls rebuild timing.
    """

    enabled: bool = False
    max_levels: int = 3
    cluster_algorithm: str = "kmeans"
    clusters_per_level: int = 10
    min_cluster_size: int = 5
    summary_model: LanguageModelClient | None = None
    summary_max_tokens: int = 256

    def __post_init__(self) -> None:
        if not (1 <= self.max_levels <= 10):
            raise ConfigurationError(f"RaptorConfig.max_levels must be in [1, 10], got {self.max_levels}")
        # Allowlist over an open string field: prevents typos becoming silent
        # fall-throughs at build time. Mirrors the iterative ``gate_mode``
        # allowlist pattern.
        if self.cluster_algorithm not in _VALID_CLUSTER_ALGORITHMS:
            raise ConfigurationError(
                f"RaptorConfig.cluster_algorithm must be one of "
                f"{sorted(_VALID_CLUSTER_ALGORITHMS)}, got {self.cluster_algorithm!r}"
            )
        if not (2 <= self.clusters_per_level <= 100):
            raise ConfigurationError(
                f"RaptorConfig.clusters_per_level must be in [2, 100], got {self.clusters_per_level}"
            )
        if not (2 <= self.min_cluster_size <= 100):
            raise ConfigurationError(f"RaptorConfig.min_cluster_size must be in [2, 100], got {self.min_cluster_size}")
        if not (50 <= self.summary_max_tokens <= 2000):
            raise ConfigurationError(
                f"RaptorConfig.summary_max_tokens must be in [50, 2000], got {self.summary_max_tokens}"
            )
        # Cross-field rule: enabling RAPTOR without a summary model would
        # silently no-op at first ``SummarizeCluster`` call. Reject at
        # construction so the misconfig surfaces at engine init.
        if self.enabled and self.summary_model is None:
            raise ConfigurationError("RaptorConfig.enabled=True requires summary_model")
