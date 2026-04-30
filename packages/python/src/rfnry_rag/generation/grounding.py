from baml_py import errors as baml_errors

from rfnry_rag.baml.baml_client.async_client import b
from rfnry_rag.common.logging import get_logger
from rfnry_rag.generation.models import RelevanceResult
from rfnry_rag.models import RetrievedChunk
from rfnry_rag.providers import LanguageModelClient, build_registry

logger = get_logger("generation/grounding")

DEFAULT_ESCALATION = (
    "I couldn't find specific information about this in the available knowledge sources. "
    "The retrieved content does not contain enough relevant information to provide a grounded answer."
)


class ScoreGate:
    """Simple score-threshold grounding gate."""

    def __init__(self, threshold: float = 0.5) -> None:
        self._threshold = threshold

    def check(self, results: list[RetrievedChunk]) -> tuple[bool, str | None]:
        if not results:
            return False, DEFAULT_ESCALATION

        best_score = max(r.score for r in results)
        if best_score < self._threshold:
            return False, DEFAULT_ESCALATION

        return True, None


class RelevanceGate:
    """Semantic relevance gate using BAML CheckRelevance to judge retrieval quality.

    Falls back to ScoreGate on failure.
    """

    def __init__(self, lm_client: LanguageModelClient, fallback_gate: ScoreGate) -> None:
        self._registry = build_registry(lm_client)
        self._fallback = fallback_gate

    async def check(
        self,
        query: str,
        results: list[RetrievedChunk],
    ) -> tuple[bool, str | None, RelevanceResult | None]:
        """Check if retrieved results can answer the query.

        Returns (passed, escalation_message, relevance_result).
        """
        if not results:
            return (
                False,
                DEFAULT_ESCALATION,
                RelevanceResult(answerable=False, confidence=0.0, relevant_indices=[]),
            )

        passages = "\n".join(f"[{i}] {r.content}" for i, r in enumerate(results))

        try:
            result = await b.CheckRelevance(
                query,
                passages,
                baml_options={"client_registry": self._registry},
            )

            relevance = RelevanceResult(
                answerable=bool(result.relevant),
                confidence=1.0 if result.relevant else 0.0,
                relevant_indices=list(range(len(results))) if result.relevant else [],
                needs_clarification=bool(result.needs_clarification),
                clarifying_question=result.clarifying_question,
                clarifying_options=result.clarifying_options,
            )

            logger.info(
                "relevance check: relevant=%s, reasoning=%s",
                result.relevant,
                (result.reasoning[:80] + "...") if len(result.reasoning) > 80 else result.reasoning,
            )

            if relevance.answerable:
                return True, None, relevance

            return False, DEFAULT_ESCALATION, relevance

        except baml_errors.BamlValidationError as exc:
            logger.exception(
                "CheckRelevance failed: LLM returned an unparseable response — falling back to score gate. Detail: %s",
                exc,
            )
            return self._fallback_check(results)
        except Exception as exc:
            logger.exception(
                "CheckRelevance failed: %s — falling back to score gate",
                exc,
            )
            return self._fallback_check(results)

    def _fallback_check(
        self,
        results: list[RetrievedChunk],
    ) -> tuple[bool, str | None, RelevanceResult | None]:
        passed, message = self._fallback.check(results)
        if passed:
            return (
                True,
                None,
                RelevanceResult(answerable=True, confidence=0.0, relevant_indices=list(range(len(results)))),
            )
        return (
            False,
            message,
            RelevanceResult(answerable=False, confidence=0.0, relevant_indices=[]),
        )
