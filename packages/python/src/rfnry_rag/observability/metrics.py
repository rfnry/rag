import asyncio
from collections import Counter
from typing import Protocol

from rfnry_rag.baml.baml_client.async_client import b
from rfnry_rag.logging import get_logger
from rfnry_rag.observability.models import MetricResult
from rfnry_rag.observability.normalize import normalize_answer
from rfnry_rag.providers import LanguageModelClient, build_registry

logger = get_logger("evaluation")


class BaseMetric(Protocol):
    """Protocol for answer-quality metrics."""

    name: str

    def score(self, prediction: str, references: list[str]) -> float: ...

    def score_batch(self, predictions: list[str], references: list[list[str]]) -> MetricResult: ...


class ExactMatch:
    """Normalized exact string match against any reference answer."""

    name: str = "em"

    def score(self, prediction: str, references: list[str]) -> float:
        normalized_pred = normalize_answer(prediction)
        for ref in references:
            if normalize_answer(ref) == normalized_pred:
                return 1.0
        return 0.0

    def score_batch(self, predictions: list[str], references: list[list[str]]) -> MetricResult:
        scores = [self.score(pred, refs) for pred, refs in zip(predictions, references, strict=True)]
        return MetricResult(mean=sum(scores) / len(scores) if scores else 0.0, scores=scores)


class F1Score:
    """Token-level F1 overlap between prediction and best-matching reference."""

    name: str = "f1"

    def _token_f1(self, prediction: str, reference: str) -> float:
        pred_tokens = normalize_answer(prediction).split()
        ref_tokens = normalize_answer(reference).split()
        if not pred_tokens or not ref_tokens:
            return 0.0

        common = Counter(pred_tokens) & Counter(ref_tokens)
        num_common = sum(common.values())
        if num_common == 0:
            return 0.0

        precision = num_common / len(pred_tokens)
        recall = num_common / len(ref_tokens)
        return (2 * precision * recall) / (precision + recall)

    def score(self, prediction: str, references: list[str]) -> float:
        return max(self._token_f1(prediction, ref) for ref in references) if references else 0.0

    def score_batch(self, predictions: list[str], references: list[list[str]]) -> MetricResult:
        scores = [self.score(pred, refs) for pred, refs in zip(predictions, references, strict=True)]
        return MetricResult(mean=sum(scores) / len(scores) if scores else 0.0, scores=scores)


class LLMJudgment:
    """LLM-as-judge metric — rates answer quality via BAML JudgeAnswerQuality."""

    name: str = "llm_judge"

    def __init__(self, lm_client: LanguageModelClient) -> None:
        self._lm_client = lm_client

    async def score(self, prediction: str, references: list[str], query: str = "") -> float:
        registry = build_registry(self._lm_client)
        best_score = 0.0

        # SERIAL: best_score is updated after each reference so a failure on one
        # reference is isolated via try/except without losing prior scores.
        # score_batch() already gathers across predictions — parallelising refs
        # here would add concurrency at both levels simultaneously.
        for ref in references:
            try:
                result = await b.JudgeAnswerQuality(
                    query=query,
                    prediction=prediction,
                    reference=ref,
                    baml_options={"client_registry": registry},
                )
                best_score = max(best_score, min(max(result.score, 0.0), 1.0))
            except Exception:
                logger.exception("JudgeAnswerQuality failed for reference — skipping")

        return best_score

    async def score_batch(
        self, predictions: list[str], references: list[list[str]], queries: list[str] | None = None
    ) -> MetricResult:
        if queries is None:
            queries = [""] * len(predictions)
        coros = [
            self.score(pred, refs, query=q) for pred, refs, q in zip(predictions, references, queries, strict=True)
        ]
        scores = list(await asyncio.gather(*coros))
        return MetricResult(mean=sum(scores) / len(scores) if scores else 0.0, scores=scores)
