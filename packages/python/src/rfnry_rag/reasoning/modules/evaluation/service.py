from __future__ import annotations

from rfnry_rag.reasoning.common.concurrency import run_concurrent
from rfnry_rag.reasoning.common.errors import EvaluationError
from rfnry_rag.reasoning.common.language_model import LanguageModelClient, build_registry
from rfnry_rag.reasoning.modules.evaluation.metrics import cosine_similarity, llm_judge, semantic_similarity
from rfnry_rag.reasoning.modules.evaluation.models import (
    EvaluationConfig,
    EvaluationPair,
    EvaluationReport,
    EvaluationResult,
)
from rfnry_rag.reasoning.protocols import BaseEmbeddings


class EvaluationService:
    """Evaluate generated outputs against reference texts."""

    def __init__(
        self,
        embeddings: BaseEmbeddings | None = None,
        lm_client: LanguageModelClient | None = None,
    ) -> None:
        self._embeddings = embeddings
        self._registry = build_registry(lm_client) if lm_client else None

    def _compute_quality_band(self, score: float, cfg: EvaluationConfig) -> str:
        if score >= cfg.high_threshold:
            return "high"
        if score >= cfg.medium_threshold:
            return "medium"
        return "low"

    async def evaluate(
        self,
        pair: EvaluationPair,
        config: EvaluationConfig | None = None,
    ) -> EvaluationResult:
        """Evaluate a single generated/reference pair."""
        cfg = config or EvaluationConfig()

        similarity = None
        judge_score = None
        judge_reasoning = None
        dimension_scores = None

        if cfg.strategy in ("similarity", "combined"):
            if not self._embeddings:
                raise EvaluationError("Similarity strategy requires embeddings")
            similarity = await semantic_similarity(pair.generated, pair.reference, self._embeddings)

        if cfg.strategy in ("judge", "combined"):
            if not self._registry:
                raise EvaluationError("Judge strategy requires lm_client")
            dim_strings = [d.name for d in cfg.dimensions] if cfg.dimensions else []
            judge_score, judge_reasoning, dimension_scores = await llm_judge(
                pair.generated,
                pair.reference,
                self._registry,
                dim_strings,
                pair.context,
                cfg.max_text_length,
            )

        if judge_score is not None:
            score = judge_score
        elif similarity is not None:
            score = similarity
        else:
            raise EvaluationError("No scoring method available for the configured strategy")

        return EvaluationResult(
            score=score,
            similarity=similarity,
            judge_score=judge_score,
            judge_reasoning=judge_reasoning,
            dimension_scores=dimension_scores,
            quality_band=self._compute_quality_band(score, cfg),
        )

    async def evaluate_batch(
        self,
        pairs: list[EvaluationPair],
        config: EvaluationConfig | None = None,
    ) -> EvaluationReport:
        """Evaluate multiple pairs concurrently and aggregate results."""
        cfg = config or EvaluationConfig()

        if cfg.strategy in ("similarity", "combined"):
            if not self._embeddings:
                raise EvaluationError("Similarity strategy requires embeddings")
            all_texts = []
            for pair in pairs:
                all_texts.append(pair.generated)
                all_texts.append(pair.reference)
            all_vectors = await self._embeddings.embed(all_texts)
        else:
            all_vectors = None

        dim_strings = [d.name for d in cfg.dimensions] if cfg.dimensions else []

        async def _evaluate_one(idx: int) -> EvaluationResult:
            pair = pairs[idx]
            similarity = None
            judge_score = None
            judge_reasoning = None
            dimension_scores = None

            if all_vectors is not None:
                similarity = cosine_similarity(all_vectors[2 * idx], all_vectors[2 * idx + 1])

            if cfg.strategy in ("judge", "combined") and self._registry:
                judge_score, judge_reasoning, dimension_scores = await llm_judge(
                    pair.generated,
                    pair.reference,
                    self._registry,
                    dim_strings,
                    pair.context,
                    cfg.max_text_length,
                )

            score = judge_score if judge_score is not None else (similarity if similarity is not None else 0.0)

            return EvaluationResult(
                score=score,
                similarity=similarity,
                judge_score=judge_score,
                judge_reasoning=judge_reasoning,
                dimension_scores=dimension_scores,
                quality_band=self._compute_quality_band(score, cfg),
            )

        results = await run_concurrent(range(len(pairs)), _evaluate_one, cfg.concurrency)

        sum_similarity = 0.0
        count_similarity = 0
        sum_judge = 0.0
        count_judge = 0
        distribution = {"high": 0, "medium": 0, "low": 0}
        for r in results:
            if r.similarity is not None:
                sum_similarity += r.similarity
                count_similarity += 1
            if r.judge_score is not None:
                sum_judge += r.judge_score
                count_judge += 1
            if r.quality_band:
                distribution[r.quality_band] += 1

        mean_similarity = sum_similarity / count_similarity if count_similarity else 0.0
        mean_judge = sum_judge / count_judge if count_judge else None

        return EvaluationReport(
            results=results,
            mean_similarity=mean_similarity,
            mean_judge_score=mean_judge,
            distribution=distribution,
        )
