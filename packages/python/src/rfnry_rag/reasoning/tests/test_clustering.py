from __future__ import annotations

import hashlib
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from rfnry_rag.reasoning.common.errors import ClusteringError
from rfnry_rag.reasoning.common.language_model import LanguageModelClient, LanguageModelProvider
from rfnry_rag.reasoning.modules.clustering.algorithms import run_clustering
from rfnry_rag.reasoning.modules.clustering.comparison import ClusterComparison, compare_clusters
from rfnry_rag.reasoning.modules.clustering.models import Cluster, ClusteringConfig, ClusteringResult, TextWithMetadata
from rfnry_rag.reasoning.modules.clustering.service import ClusteringService


class _MockEmbeddings:
    model = "test-model"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            h = hashlib.md5(text.encode()).hexdigest()
            vectors.append([int(c, 16) / 15.0 for c in h[:16]])
        return vectors

    async def embedding_dimension(self) -> int:
        return 16


def _lm_client() -> LanguageModelClient:
    return LanguageModelClient(
        provider=LanguageModelProvider(provider="openai", model="gpt-4o-mini", api_key="test-key"),
    )


def _make_cluster(cluster_id: int, size: int, centroid: list[float], label: str | None = None) -> Cluster:
    return Cluster(
        cluster_id=cluster_id,
        label=label,
        size=size,
        percentage=0.0,
        centroid=centroid,
        sample_texts=[f"sample-{cluster_id}"],
    )


def test_kmeans_clusters_vectors():
    vectors = np.array(
        [[0.0, 0.0], [0.1, 0.1], [10.0, 10.0], [10.1, 10.1]],
        dtype=np.float32,
    )
    config = ClusteringConfig(algorithm="kmeans", n_clusters=2, random_state=42)
    labels, centroids = run_clustering(vectors, config)
    assert labels.shape == (4,)
    assert centroids.shape == (2, 2)
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]


def test_kmeans_caps_n_clusters():
    vectors = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    config = ClusteringConfig(algorithm="kmeans", n_clusters=100, random_state=42)
    labels, centroids = run_clustering(vectors, config)
    assert centroids.shape[0] == 2


def test_clustering_empty_dataset_raises():
    vectors = np.empty((0, 3), dtype=np.float32)
    config = ClusteringConfig(algorithm="kmeans", n_clusters=2)
    with pytest.raises(ClusteringError, match="empty dataset"):
        run_clustering(vectors, config)


def test_clustering_unknown_algorithm_raises():
    vectors = np.array([[1.0]], dtype=np.float32)
    config = ClusteringConfig.__new__(ClusteringConfig)
    config.algorithm = "invalid"
    config.n_clusters = 2
    config.min_cluster_size = 10
    config.samples_per_cluster = 5
    config.generate_labels = False
    config.random_state = 42
    with pytest.raises(ClusteringError, match="Unknown algorithm"):
        run_clustering(vectors, config)


def test_hdbscan_clusters_vectors():
    rng = np.random.RandomState(42)
    group_a = rng.randn(30, 2).astype(np.float32)
    group_b = rng.randn(30, 2).astype(np.float32) + 20
    vectors = np.vstack([group_a, group_b])
    config = ClusteringConfig(algorithm="hdbscan", min_cluster_size=5)
    labels, centroids = run_clustering(vectors, config)
    assert labels.shape == (60,)
    unique = set(labels) - {-1}
    assert len(unique) >= 2
    assert centroids.shape[0] >= 2


async def test_cluster_texts_plain():
    service = ClusteringService(embeddings=_MockEmbeddings())
    texts = [f"text about topic {i % 3}" for i in range(12)]
    result = await service.cluster_texts(texts, config=ClusteringConfig(n_clusters=3))
    assert isinstance(result, ClusteringResult)
    assert result.total_documents == 12
    assert result.algorithm == "kmeans"
    assert len(result.clusters) == 3
    assert sum(c.size for c in result.clusters) == 12


async def test_cluster_texts_with_metadata():
    service = ClusteringService(embeddings=_MockEmbeddings())
    items = [TextWithMetadata(text=f"item {i}", metadata={"idx": i}) for i in range(6)]
    result = await service.cluster_texts(items, config=ClusteringConfig(n_clusters=2))
    assert result.total_documents == 6
    has_metadata = any(c.sample_metadata for c in result.clusters)
    assert has_metadata


async def test_cluster_texts_too_few_raises():
    service = ClusteringService(embeddings=_MockEmbeddings())
    with pytest.raises(ClusteringError, match="At least 2"):
        await service.cluster_texts(["only one"])


async def test_cluster_knowledge():
    embeddings = _MockEmbeddings()

    async def mock_scroll(filters: dict | None = None, limit: int = 100, offset: str | None = None):
        if offset is not None:
            return [], None
        docs = [SimpleNamespace(payload={"text": f"doc {i}", "id": f"id-{i}"}, vector=None) for i in range(8)]
        return docs, None

    vector_store = SimpleNamespace(scroll=mock_scroll)
    service = ClusteringService(embeddings=embeddings)
    result = await service.cluster_knowledge(vector_store, "kb-1", config=ClusteringConfig(n_clusters=2))
    assert result.total_documents == 8
    assert len(result.clusters) == 2


async def test_cluster_knowledge_too_few_raises():
    async def mock_scroll(**_: Any):
        return [SimpleNamespace(payload={"text": "only", "id": "1"}, vector=None)], None

    vector_store = SimpleNamespace(scroll=mock_scroll)
    service = ClusteringService(embeddings=_MockEmbeddings())
    with pytest.raises(ClusteringError, match="At least 2"):
        await service.cluster_knowledge(vector_store, "kb-1")


@patch("rfnry_rag.reasoning.modules.clustering.labeling.b")
async def test_cluster_texts_with_labels(mock_b):
    mock_b.LabelCluster = AsyncMock(return_value=SimpleNamespace(label="auto-label"))
    service = ClusteringService(embeddings=_MockEmbeddings(), lm_client=_lm_client())
    texts = [f"text {i}" for i in range(6)]
    result = await service.cluster_texts(texts, config=ClusteringConfig(n_clusters=2, generate_labels=True))
    assert all(c.label == "auto-label" for c in result.clusters)


def test_compare_matched_growing():
    centroid = [1.0, 0.0, 0.0]
    prev = ClusteringResult(
        clusters=[_make_cluster(0, 10, centroid, "A")],
        total_documents=10,
        algorithm="kmeans",
    )
    curr = ClusteringResult(
        clusters=[_make_cluster(0, 15, centroid, "A")],
        total_documents=15,
        algorithm="kmeans",
    )
    comp = compare_clusters(prev, curr)
    assert len(comp.growing_clusters) == 1
    assert comp.growing_clusters[0].size_delta == 5


def test_compare_matched_shrinking():
    centroid = [0.0, 1.0, 0.0]
    prev = ClusteringResult(
        clusters=[_make_cluster(0, 20, centroid)],
        total_documents=20,
        algorithm="kmeans",
    )
    curr = ClusteringResult(
        clusters=[_make_cluster(0, 10, centroid)],
        total_documents=10,
        algorithm="kmeans",
    )
    comp = compare_clusters(prev, curr)
    assert len(comp.shrinking_clusters) == 1


def test_compare_matched_stable():
    centroid = [0.0, 0.0, 1.0]
    prev = ClusteringResult(
        clusters=[_make_cluster(0, 100, centroid)],
        total_documents=100,
        algorithm="kmeans",
    )
    curr = ClusteringResult(
        clusters=[_make_cluster(0, 105, centroid)],
        total_documents=105,
        algorithm="kmeans",
    )
    comp = compare_clusters(prev, curr)
    assert len(comp.stable_clusters) == 1


def test_compare_new_cluster():
    prev = ClusteringResult(clusters=[], total_documents=0, algorithm="kmeans")
    curr = ClusteringResult(
        clusters=[_make_cluster(0, 10, [1.0, 0.0])],
        total_documents=10,
        algorithm="kmeans",
    )
    comp = compare_clusters(prev, curr)
    assert len(comp.new_clusters) == 1
    assert comp.new_clusters[0].change_type == "new"


def test_compare_disappeared_cluster():
    prev = ClusteringResult(
        clusters=[_make_cluster(0, 10, [1.0, 0.0])],
        total_documents=10,
        algorithm="kmeans",
    )
    curr = ClusteringResult(clusters=[], total_documents=0, algorithm="kmeans")
    comp = compare_clusters(prev, curr)
    assert len(comp.disappeared_clusters) == 1
    assert comp.disappeared_clusters[0].change_type == "disappeared"


def test_compare_empty_both():
    prev = ClusteringResult(clusters=[], total_documents=0, algorithm="kmeans")
    curr = ClusteringResult(clusters=[], total_documents=0, algorithm="kmeans")
    comp = compare_clusters(prev, curr)
    assert isinstance(comp, ClusterComparison)
    assert not comp.new_clusters
    assert not comp.disappeared_clusters


def test_config_unknown_algorithm_raises():
    with pytest.raises(ValueError, match="Unknown algorithm"):
        ClusteringConfig(algorithm="invalid")


def test_config_n_clusters_below_2_raises():
    with pytest.raises(ValueError, match="n_clusters must be >= 2"):
        ClusteringConfig(algorithm="kmeans", n_clusters=1)


def test_config_samples_per_cluster_below_1_raises():
    with pytest.raises(ValueError, match="samples_per_cluster"):
        ClusteringConfig(samples_per_cluster=0)


def test_cluster_to_dict_minimal():
    c = Cluster(cluster_id=0, size=5, percentage=0.5, sample_texts=["a", "b"])
    d = c.to_dict()
    assert d["cluster_id"] == 0
    assert d["size"] == 5
    assert "label" not in d
    assert "sample_ids" not in d


def test_cluster_to_dict_full():
    c = Cluster(
        cluster_id=1,
        label="topic-a",
        size=10,
        percentage=0.25,
        sample_texts=["x"],
        sample_ids=["id-1"],
        sample_metadata=[{"key": "val"}],
    )
    d = c.to_dict()
    assert d["label"] == "topic-a"
    assert d["sample_ids"] == ["id-1"]
    assert d["sample_metadata"] == [{"key": "val"}]
