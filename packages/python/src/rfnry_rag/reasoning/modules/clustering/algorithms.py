from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans

from rfnry_rag.reasoning.common.errors import ClusteringError
from rfnry_rag.reasoning.modules.clustering.models import ClusteringConfig


def run_clustering(
    vectors: NDArray[np.float32],
    config: ClusteringConfig,
) -> tuple[NDArray[np.int32], NDArray[np.float32]]:
    if vectors.shape[0] == 0:
        raise ClusteringError("Cannot cluster an empty dataset")

    if config.algorithm == "kmeans":
        return _run_kmeans(vectors, config)
    elif config.algorithm == "hdbscan":
        return _run_hdbscan(vectors, config)
    else:
        raise ClusteringError(f"Unknown algorithm: {config.algorithm}")


def _run_kmeans(
    vectors: NDArray[np.float32],
    config: ClusteringConfig,
) -> tuple[NDArray[np.int32], NDArray[np.float32]]:
    n_clusters = min(config.n_clusters, vectors.shape[0])
    km = KMeans(n_clusters=n_clusters, random_state=config.random_state, n_init=10)
    labels = km.fit_predict(vectors)
    return labels.astype(np.int32), km.cluster_centers_.astype(np.float32)


def _run_hdbscan(
    vectors: NDArray[np.float32],
    config: ClusteringConfig,
) -> tuple[NDArray[np.int32], NDArray[np.float32]]:
    try:
        from sklearn.cluster import HDBSCAN
    except ImportError as exc:
        raise ClusteringError(
            "HDBSCAN requires scikit-learn >= 1.3. Install with: pip install 'scikit-learn>=1.3'"
        ) from exc

    hdb = HDBSCAN(min_cluster_size=config.min_cluster_size)
    labels = hdb.fit_predict(vectors).astype(np.int32)

    unique_labels = set(labels)
    unique_labels.discard(-1)
    centroids = []
    for label in sorted(unique_labels):
        mask = labels == label
        centroids.append(vectors[mask].mean(axis=0))

    centroid_array = (
        np.array(centroids, dtype=np.float32) if centroids else np.empty((0, vectors.shape[1]), dtype=np.float32)
    )
    return labels, centroid_array
