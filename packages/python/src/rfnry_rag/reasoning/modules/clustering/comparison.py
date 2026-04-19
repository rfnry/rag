"""Cluster comparison — detect trends between two clustering periods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from rfnry_rag.reasoning.modules.clustering.models import ClusteringResult


@dataclass
class ClusterChange:
    """Describes a change in a single cluster between periods."""

    label: str | None
    cluster_id: int
    previous_size: int
    current_size: int
    change_type: Literal["new", "growing", "shrinking", "disappeared", "stable"]
    size_delta: int
    percentage_delta: float


@dataclass
class ClusterComparison:
    """Full comparison between two clustering periods."""

    new_clusters: list[ClusterChange]
    growing_clusters: list[ClusterChange]
    shrinking_clusters: list[ClusterChange]
    disappeared_clusters: list[ClusterChange]
    stable_clusters: list[ClusterChange]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    dot = float(np.dot(va, vb))
    norm = float(np.linalg.norm(va) * np.linalg.norm(vb))
    return dot / norm if norm > 0 else 0.0


def compare_clusters(
    previous: ClusteringResult,
    current: ClusteringResult,
    similarity_threshold: float = 0.8,
) -> ClusterComparison:
    """Compare two clustering results to detect trends.

    Matches clusters between periods using centroid cosine similarity.
    Greedy matching: pair clusters with highest similarity above threshold.
    """
    prev_clusters = previous.clusters
    curr_clusters = current.clusters

    if not prev_clusters and not curr_clusters:
        return ClusterComparison([], [], [], [], [])

    similarities: list[tuple[float, int, int]] = []
    for pi, pc in enumerate(prev_clusters):
        for ci, cc in enumerate(curr_clusters):
            if pc.centroid and cc.centroid:
                sim = _cosine_similarity(pc.centroid, cc.centroid)
                similarities.append((sim, pi, ci))

    similarities.sort(reverse=True)
    matched_prev: set[int] = set()
    matched_curr: set[int] = set()
    matches: list[tuple[int, int]] = []

    for sim, pi, ci in similarities:
        if sim < similarity_threshold:
            break
        if pi in matched_prev or ci in matched_curr:
            continue
        matches.append((pi, ci))
        matched_prev.add(pi)
        matched_curr.add(ci)

    new_clusters: list[ClusterChange] = []
    growing: list[ClusterChange] = []
    shrinking: list[ClusterChange] = []
    disappeared: list[ClusterChange] = []
    stable: list[ClusterChange] = []

    growth_threshold = 0.1
    for pi, ci in matches:
        pc = prev_clusters[pi]
        cc = curr_clusters[ci]
        delta = cc.size - pc.size
        pct_delta = delta / pc.size if pc.size > 0 else 0.0

        if abs(pct_delta) <= growth_threshold:
            change_type: Literal["growing", "shrinking", "stable"] = "stable"
        elif delta > 0:
            change_type = "growing"
        else:
            change_type = "shrinking"

        change = ClusterChange(
            label=cc.label or pc.label,
            cluster_id=cc.cluster_id,
            previous_size=pc.size,
            current_size=cc.size,
            change_type=change_type,
            size_delta=delta,
            percentage_delta=round(pct_delta, 4),
        )

        if change_type == "growing":
            growing.append(change)
        elif change_type == "shrinking":
            shrinking.append(change)
        else:
            stable.append(change)

    for ci, cc in enumerate(curr_clusters):
        if ci not in matched_curr:
            new_clusters.append(
                ClusterChange(
                    label=cc.label,
                    cluster_id=cc.cluster_id,
                    previous_size=0,
                    current_size=cc.size,
                    change_type="new",
                    size_delta=cc.size,
                    percentage_delta=0.0,
                )
            )

    for pi, pc in enumerate(prev_clusters):
        if pi not in matched_prev:
            disappeared.append(
                ClusterChange(
                    label=pc.label,
                    cluster_id=pc.cluster_id,
                    previous_size=pc.size,
                    current_size=0,
                    change_type="disappeared",
                    size_delta=-pc.size,
                    percentage_delta=-1.0,
                )
            )

    return ClusterComparison(
        new_clusters=new_clusters,
        growing_clusters=growing,
        shrinking_clusters=shrinking,
        disappeared_clusters=disappeared,
        stable_clusters=stable,
    )
