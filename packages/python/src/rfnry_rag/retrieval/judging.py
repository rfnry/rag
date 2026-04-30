from typing import Protocol

from baml_py import errors as baml_errors

from rfnry_rag.baml.baml_client.async_client import b
from rfnry_rag.logging import get_logger
from rfnry_rag.observability.models import JudgmentResult
from rfnry_rag.providers import LanguageModelClient, build_registry

logger = get_logger("retrieval/judging")


class BaseRetrievalJudgment(Protocol):
    """Protocol for retrieval necessity judges."""

    async def should_retrieve(self, query: str) -> JudgmentResult: ...


class RetrievalJudgment:
    """LLM-based retrieval necessity judge via BAML JudgeRetrievalNecessity.

    Classifies whether a query needs domain-specific retrieval or can be
    answered from general knowledge alone.
    """

    def __init__(
        self,
        lm_client: LanguageModelClient,
        knowledge_description: str | None = None,
    ) -> None:
        self._lm_client = lm_client
        self._knowledge_description = knowledge_description or "A domain-specific knowledge base."

    async def should_retrieve(self, query: str) -> JudgmentResult:
        registry = build_registry(self._lm_client)

        try:
            result = await b.JudgeRetrievalNecessity(
                query=query,
                knowledge_description=self._knowledge_description,
                baml_options={"client_registry": registry},
            )

            logger.info(
                "retrieval judge: should_retrieve=%s, confidence=%.2f, reasoning=%s",
                result.should_retrieve,
                result.confidence,
                (result.reasoning[:80] + "...") if len(result.reasoning) > 80 else result.reasoning,
            )

            return JudgmentResult(
                should_retrieve=bool(result.should_retrieve),
                confidence=min(max(result.confidence, 0.0), 1.0),
                reasoning=result.reasoning,
            )

        except baml_errors.BamlValidationError as exc:
            logger.exception(
                "JudgeRetrievalNecessity failed: LLM returned unparseable response — "
                "defaulting to should_retrieve=True. Detail: %s",
                exc,
            )
            return JudgmentResult(
                should_retrieve=True, confidence=0.0, reasoning="Judge failed — defaulting to retrieve"
            )

        except Exception as exc:
            logger.exception("JudgeRetrievalNecessity failed: %s — defaulting to should_retrieve=True", exc)
            return JudgmentResult(
                should_retrieve=True, confidence=0.0, reasoning="Judge failed — defaulting to retrieve"
            )
