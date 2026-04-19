from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class TextWithMetadata:
    """A text with optional metadata for clustering."""

    text: str
    metadata: dict[str, Any] | None = None


@dataclass
class Cluster:
    """A single cluster discovered in a corpus."""

    cluster_id: int
    label: str | None = None
    size: int = 0
    percentage: float = 0.0
    centroid: list[float] = field(default_factory=list)
    sample_texts: list[str] = field(default_factory=list)
    sample_ids: list[str] = field(default_factory=list)
    sample_metadata: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "cluster_id": self.cluster_id,
            "size": self.size,
            "percentage": self.percentage,
            "sample_texts": self.sample_texts,
        }
        if self.label is not None:
            d["label"] = self.label
        if self.sample_ids:
            d["sample_ids"] = self.sample_ids
        if self.sample_metadata:
            d["sample_metadata"] = self.sample_metadata
        return d


@dataclass
class ClusteringConfig:
    """Configuration for clustering operations."""

    algorithm: Literal["kmeans", "hdbscan"] = "kmeans"
    n_clusters: int = 10
    min_cluster_size: int = 10
    samples_per_cluster: int = 5
    generate_labels: bool = False
    random_state: int = 42

    def __post_init__(self) -> None:
        if self.algorithm not in ("kmeans", "hdbscan"):
            raise ValueError(f"Unknown algorithm: {self.algorithm}. Must be 'kmeans' or 'hdbscan'.")
        if self.algorithm == "kmeans" and self.n_clusters < 2:
            raise ValueError("n_clusters must be >= 2 for kmeans")
        if self.samples_per_cluster < 1:
            raise ValueError("samples_per_cluster must be >= 1")


@dataclass
class ClusteringResult:
    """Full result of a clustering operation."""

    clusters: list[Cluster]
    total_documents: int
    algorithm: Literal["kmeans", "hdbscan"]
    noise_count: int = 0
