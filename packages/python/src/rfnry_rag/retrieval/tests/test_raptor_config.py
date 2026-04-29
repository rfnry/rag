"""RaptorConfig dataclass invariants.

Tests pin the dataclass defaults, bounds, allowlist, and the cross-field
rule (``enabled=True`` requires ``summary_model``).
"""

from __future__ import annotations

import pytest

from rfnry_rag.retrieval.common.errors import ConfigurationError
from rfnry_rag.retrieval.common.language_model import LanguageModelClient, LanguageModelProvider
from rfnry_rag.retrieval.modules.ingestion.methods.raptor.config import RaptorConfig


def _lm_client() -> LanguageModelClient:
    return LanguageModelClient(
        provider=LanguageModelProvider(provider="openai", model="gpt-test", api_key="x"),
    )


def test_raptor_config_defaults_are_off() -> None:
    cfg = RaptorConfig()
    assert cfg.enabled is False
    assert cfg.max_levels == 3
    assert cfg.cluster_algorithm == "kmeans"
    assert cfg.clusters_per_level == 10
    assert cfg.min_cluster_size == 5
    assert cfg.summary_max_tokens == 256
    assert cfg.summary_model is None


def test_raptor_config_max_levels_bounds() -> None:
    # Out-of-range raises, boundary values accepted.
    with pytest.raises(ConfigurationError, match="max_levels"):
        RaptorConfig(max_levels=0)
    with pytest.raises(ConfigurationError, match="max_levels"):
        RaptorConfig(max_levels=11)
    assert RaptorConfig(max_levels=1).max_levels == 1
    assert RaptorConfig(max_levels=10).max_levels == 10


def test_raptor_config_cluster_algorithm_allowlist() -> None:
    with pytest.raises(ConfigurationError, match="cluster_algorithm"):
        RaptorConfig(cluster_algorithm="bogus")
    assert RaptorConfig(cluster_algorithm="kmeans").cluster_algorithm == "kmeans"
    assert RaptorConfig(cluster_algorithm="hdbscan").cluster_algorithm == "hdbscan"


def test_raptor_config_clusters_per_level_bounds() -> None:
    with pytest.raises(ConfigurationError, match="clusters_per_level"):
        RaptorConfig(clusters_per_level=1)
    with pytest.raises(ConfigurationError, match="clusters_per_level"):
        RaptorConfig(clusters_per_level=101)
    assert RaptorConfig(clusters_per_level=2).clusters_per_level == 2
    assert RaptorConfig(clusters_per_level=100).clusters_per_level == 100


def test_raptor_config_min_cluster_size_bounds() -> None:
    with pytest.raises(ConfigurationError, match="min_cluster_size"):
        RaptorConfig(min_cluster_size=1)
    with pytest.raises(ConfigurationError, match="min_cluster_size"):
        RaptorConfig(min_cluster_size=101)
    assert RaptorConfig(min_cluster_size=2).min_cluster_size == 2
    assert RaptorConfig(min_cluster_size=100).min_cluster_size == 100


def test_raptor_config_summary_max_tokens_bounds() -> None:
    with pytest.raises(ConfigurationError, match="summary_max_tokens"):
        RaptorConfig(summary_max_tokens=49)
    with pytest.raises(ConfigurationError, match="summary_max_tokens"):
        RaptorConfig(summary_max_tokens=2001)
    assert RaptorConfig(summary_max_tokens=50).summary_max_tokens == 50
    assert RaptorConfig(summary_max_tokens=2000).summary_max_tokens == 2000


def test_raptor_config_enabled_requires_summary_model() -> None:
    # Cross-field invariant: enabling RAPTOR without a summary model would
    # silently no-op at first SummarizeCluster call. Reject at construction.
    with pytest.raises(ConfigurationError, match="summary_model"):
        RaptorConfig(enabled=True, summary_model=None)
    # With a model, enabled=True succeeds.
    cfg = RaptorConfig(enabled=True, summary_model=_lm_client())
    assert cfg.enabled is True
    assert cfg.summary_model is not None
