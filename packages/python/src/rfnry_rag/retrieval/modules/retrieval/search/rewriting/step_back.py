from dataclasses import dataclass

from baml_py import errors as baml_errors

from rfnry_rag.common.logging import query_logging_enabled
from rfnry_rag.retrieval.baml.baml_client.async_client import b
from rfnry_rag.retrieval.common.language_model import LanguageModelClient, build_registry
from rfnry_rag.retrieval.common.logging import get_logger

logger = get_logger("retrieval/rewriting/step_back")


@dataclass
class StepBackRewriting:
    """Generate a broader version of the query to retrieve contextual information.

    Produces a more general version of an overly-specific query so that search
    retrieves background context (overview sections, specs tables) alongside
    the precise matches from the original query.
    """

    lm_client: LanguageModelClient

    async def rewrite(self, query: str, conversation_context: str | None = None) -> list[str]:
        registry = build_registry(self.lm_client)
        context_hint = f"\n\nConversation context: {conversation_context}" if conversation_context else ""

        try:
            result = await b.GenerateStepBackQuery(
                query + context_hint,
                baml_options={"client_registry": registry},
            )
            if query_logging_enabled():
                logger.info("step-back: '%s' -> '%s'", query[:60], result.broader_query[:60])
            else:
                logger.info("step-back rewrite completed (orig_len=%d)", len(query))
            return [result.broader_query]

        except baml_errors.BamlValidationError as exc:
            logger.exception(
                "GenerateStepBackQuery failed: LLM returned unparseable response — "
                "falling back to original query. Detail: %s",
                exc,
            )
            return []
        except Exception as exc:
            logger.exception("GenerateStepBackQuery failed: %s — falling back to original query", exc)
            return []
