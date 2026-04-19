from __future__ import annotations

from typing import Any

from rfnry_rag.reasoning.common.concurrency import run_concurrent
from rfnry_rag.reasoning.common.errors import ClassificationError
from rfnry_rag.reasoning.common.language_model import LanguageModelClient, build_registry
from rfnry_rag.reasoning.modules.classification.models import (
    CategoryDefinition,
    Classification,
    ClassificationConfig,
    ClassificationSetDefinition,
    ClassificationSetResult,
)
from rfnry_rag.reasoning.modules.classification.strategies import (
    knn_classify,
    knn_classify_with_vector,
    llm_classify,
    llm_classify_sets,
)
from rfnry_rag.reasoning.protocols import BaseEmbeddings, BaseSemanticIndex


class ClassificationService:
    """Text classification using LLM, kNN, or hybrid strategies."""

    def __init__(
        self,
        embeddings: BaseEmbeddings | None = None,
        lm_client: LanguageModelClient | None = None,
        vector_store: BaseSemanticIndex | None = None,
    ) -> None:
        self._embeddings = embeddings
        self._registry = build_registry(lm_client) if lm_client else None
        self._vector_store = vector_store

    async def classify(
        self,
        text: str,
        categories: list[CategoryDefinition],
        config: ClassificationConfig | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Classification:
        """Classify a single text into one of the provided categories."""
        cfg = config or ClassificationConfig()
        result = await self._run_strategy(text, categories, cfg)
        result.metadata = metadata
        if cfg.low_confidence_threshold is not None and result.confidence < cfg.low_confidence_threshold:
            result.needs_review = True
        return result

    async def _run_strategy(
        self,
        text: str,
        categories: list[CategoryDefinition],
        cfg: ClassificationConfig,
    ) -> Classification:
        if cfg.strategy == "llm":
            if not self._registry:
                raise ClassificationError("LLM classification requires lm_client")
            return await llm_classify(text, categories, self._registry, cfg.max_text_length)

        if not self._embeddings:
            raise ClassificationError("Hybrid classification requires embeddings")
        if not self._vector_store:
            raise ClassificationError("Hybrid classification requires vector_store")
        if not cfg.knn_knowledge_id:
            raise ClassificationError("Hybrid classification requires knn_knowledge_id in config")

        knn_result = await knn_classify(
            text,
            self._embeddings,
            self._vector_store,
            cfg.top_k,
            cfg.knn_knowledge_id,
            cfg.knn_label_field,
        )

        if knn_result.confidence >= cfg.escalation_threshold:
            knn_result.strategy_used = "hybrid_knn"
            return knn_result

        if not self._registry:
            knn_result.strategy_used = "hybrid_knn"
            return knn_result

        llm_result = await llm_classify(text, categories, self._registry, cfg.max_text_length)
        llm_result.strategy_used = "hybrid_llm_escalation"
        return llm_result

    async def classify_batch(
        self,
        texts: list[str],
        categories: list[CategoryDefinition],
        config: ClassificationConfig | None = None,
        metadata: list[dict[str, Any]] | None = None,
    ) -> list[Classification]:
        """Classify multiple texts concurrently."""
        if metadata is not None and len(metadata) != len(texts):
            raise ClassificationError(f"metadata length ({len(metadata)}) must match texts length ({len(texts)})")

        cfg = config or ClassificationConfig()

        if cfg.strategy == "llm":

            async def _classify_one(args: tuple[str, dict[str, Any] | None]) -> Classification:
                text, meta = args
                return await self.classify(text, categories, cfg, metadata=meta)

            items = [(t, metadata[i] if metadata else None) for i, t in enumerate(texts)]
            return await run_concurrent(items, _classify_one, cfg.concurrency)

        if not self._embeddings:
            raise ClassificationError("Hybrid classification requires embeddings")
        if not self._vector_store:
            raise ClassificationError("Hybrid classification requires vector_store")
        if not cfg.knn_knowledge_id:
            raise ClassificationError("Hybrid classification requires knn_knowledge_id in config")

        all_vectors = await self._embeddings.embed(texts)
        vector_store = self._vector_store
        knowledge_id = cfg.knn_knowledge_id

        async def _classify_hybrid(idx: int) -> Classification:
            text = texts[idx]
            vector = all_vectors[idx]
            meta = metadata[idx] if metadata else None

            knn_result = await knn_classify_with_vector(
                vector,
                vector_store,
                cfg.top_k,
                knowledge_id,
                cfg.knn_label_field,
            )

            if knn_result.confidence >= cfg.escalation_threshold:
                knn_result.strategy_used = "hybrid_knn"
                knn_result.metadata = meta
                return knn_result

            if not self._registry:
                knn_result.strategy_used = "hybrid_knn"
                knn_result.metadata = meta
                return knn_result

            llm_result = await llm_classify(text, categories, self._registry, cfg.max_text_length)
            llm_result.strategy_used = "hybrid_llm_escalation"
            llm_result.metadata = meta
            return llm_result

        return await run_concurrent(range(len(texts)), _classify_hybrid, cfg.concurrency)

    async def classify_sets(
        self,
        text: str,
        sets: list[ClassificationSetDefinition],
        config: ClassificationConfig | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ClassificationSetResult:
        """Classify text against multiple category sets in one LLM call."""
        cfg = config or ClassificationConfig()
        if not self._registry:
            raise ClassificationError("Multi-set classification requires lm_client")

        classifications = await llm_classify_sets(text, sets, self._registry, cfg.max_text_length)

        for c in classifications.values():
            if cfg.low_confidence_threshold is not None and c.confidence < cfg.low_confidence_threshold:
                c.needs_review = True
            c.metadata = metadata

        return ClassificationSetResult(classifications=classifications)

    async def classify_sets_batch(
        self,
        texts: list[str],
        sets: list[ClassificationSetDefinition],
        config: ClassificationConfig | None = None,
    ) -> list[ClassificationSetResult]:
        """Classify multiple texts against category sets concurrently."""
        cfg = config or ClassificationConfig()

        async def _classify_one(text: str) -> ClassificationSetResult:
            return await self.classify_sets(text, sets, cfg)

        return await run_concurrent(texts, _classify_one, cfg.concurrency)
