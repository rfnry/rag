from __future__ import annotations

from typing import Any

import numpy as np

from rfnry_rag.common.embeddings import embed_batched
from rfnry_rag.reasoning.common.errors import ClusteringError
from rfnry_rag.reasoning.common.language_model import LanguageModelClient, build_registry
from rfnry_rag.reasoning.common.logging import get_logger
from rfnry_rag.reasoning.modules.clustering.algorithms import run_clustering
from rfnry_rag.reasoning.modules.clustering.labeling import generate_cluster_labels
from rfnry_rag.reasoning.modules.clustering.models import Cluster, ClusteringConfig, ClusteringResult, TextWithMetadata
from rfnry_rag.reasoning.protocols import BaseEmbeddings, BaseSemanticIndex

logger = get_logger("clustering")


class ClusteringService:
    """Cluster texts or knowledge base documents."""

    def __init__(
        self,
        embeddings: BaseEmbeddings,
        lm_client: LanguageModelClient | None = None,
    ) -> None:
        self._embeddings = embeddings
        self._registry = build_registry(lm_client) if lm_client else None

    async def cluster_texts(
        self,
        texts: list[str] | list[TextWithMetadata],
        config: ClusteringConfig | None = None,
    ) -> ClusteringResult:
        """Cluster a list of raw texts or TextWithMetadata objects."""
        cfg = config or ClusteringConfig()

        metadata_list: list[dict[str, Any] | None] | None = None
        if texts and isinstance(texts[0], TextWithMetadata):
            items = texts  # type: ignore[assignment]
            metadata_list = [item.metadata for item in items]  # type: ignore[union-attr]
            plain_texts = [item.text for item in items]  # type: ignore[union-attr]
        else:
            plain_texts = texts  # type: ignore[assignment]

        if len(plain_texts) < 2:
            raise ClusteringError("At least 2 texts are required for clustering")

        logger.info("[clustering/embed] embedding %d texts", len(plain_texts))
        vectors = await self._embed_batched(plain_texts)
        matrix = np.array(vectors, dtype=np.float32)

        logger.info("[clustering/%s] running clustering", cfg.algorithm)
        labels, centroids = run_clustering(matrix, cfg)
        return await self._build_result(plain_texts, None, labels, centroids, cfg, metadata_list)

    async def cluster_knowledge(
        self,
        vector_store: BaseSemanticIndex,
        knowledge_id: str,
        config: ClusteringConfig | None = None,
    ) -> ClusteringResult:
        """Cluster documents from a vector store."""
        cfg = config or ClusteringConfig()

        texts: list[str] = []
        doc_ids: list[str] = []
        vectors: list[list[float]] = []
        offset: str | None = None

        while True:
            batch, next_offset = await vector_store.scroll(
                filters={"knowledge_id": knowledge_id},
                limit=100,
                offset=offset,
            )
            if not batch:
                break
            for doc in batch:
                payload = doc.payload if hasattr(doc, "payload") else doc
                texts.append(payload.get("text", ""))
                doc_ids.append(payload.get("id", ""))
                if hasattr(doc, "vector") and doc.vector:
                    vectors.append(doc.vector)
            offset = next_offset
            if offset is None:
                break

        if len(texts) < 2:
            raise ClusteringError("At least 2 documents required for clustering")

        if vectors and len(vectors) == len(texts):
            matrix = np.array(vectors, dtype=np.float32)
        else:
            logger.info("[clustering/embed] embedding %d documents", len(texts))
            raw_vectors = await self._embed_batched(texts)
            matrix = np.array(raw_vectors, dtype=np.float32)

        logger.info("[clustering/%s] running clustering", cfg.algorithm)
        labels, centroids = run_clustering(matrix, cfg)
        return await self._build_result(texts, doc_ids, labels, centroids, cfg)

    async def _build_result(
        self,
        texts: list[str],
        doc_ids: list[str] | None,
        labels: np.ndarray,
        centroids: np.ndarray,
        config: ClusteringConfig,
        metadata_list: list[dict[str, Any] | None] | None = None,
    ) -> ClusteringResult:
        noise_count = int(np.sum(labels == -1))
        total = len(texts)

        cluster_groups: dict[int, list[int]] = {}
        for idx, label in enumerate(labels):
            if label == -1:
                continue
            cluster_groups.setdefault(int(label), []).append(idx)

        sample_groups: list[list[str]] = []
        clusters: list[Cluster] = []

        for cluster_id in sorted(cluster_groups.keys()):
            indices = cluster_groups[cluster_id]
            sample_indices = indices[: config.samples_per_cluster]
            sample_texts = [texts[i] for i in sample_indices]
            sample_ids = [doc_ids[i] for i in sample_indices] if doc_ids else []
            sample_metadata: list[dict[str, Any]] = []
            if metadata_list:
                for i in sample_indices:
                    m = metadata_list[i]
                    if m is not None:
                        sample_metadata.append(m)

            centroid = centroids[cluster_id].tolist() if cluster_id < len(centroids) else []

            clusters.append(
                Cluster(
                    cluster_id=cluster_id,
                    label=None,
                    size=len(indices),
                    percentage=len(indices) / total if total > 0 else 0.0,
                    centroid=centroid,
                    sample_texts=sample_texts,
                    sample_ids=sample_ids,
                    sample_metadata=sample_metadata,
                )
            )
            sample_groups.append(sample_texts)

        if config.generate_labels and self._registry and sample_groups:
            generated_labels = await generate_cluster_labels(sample_groups, self._registry)
            for cluster, label in zip(clusters, generated_labels, strict=True):
                cluster.label = label

        return ClusteringResult(
            clusters=clusters,
            total_documents=total,
            algorithm=config.algorithm,
            noise_count=noise_count,
        )

    async def _embed_batched(self, texts: list[str]) -> list[list[float]]:
        return await embed_batched(self._embeddings, texts)
