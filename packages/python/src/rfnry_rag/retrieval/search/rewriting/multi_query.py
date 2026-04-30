from dataclasses import dataclass

from baml_py import errors as baml_errors

from rfnry_rag.baml.baml_client.async_client import b
from rfnry_rag.common.language_model import LanguageModelClient, build_registry
from rfnry_rag.common.logging import get_logger
from rfnry_rag.exceptions import ConfigurationError

logger = get_logger("retrieval/rewriting/multi_query")

DEFAULT_NUM_VARIANTS = 3


@dataclass
class MultiQueryRewriting:
    """Generate multiple query variants to improve recall.

    Produces 2-3 alternative formulations of the original query, each
    approaching the topic from a different angle or using different keywords.
    All variants are searched alongside the original, and results are merged
    via reciprocal rank fusion.
    """

    lm_client: LanguageModelClient
    num_variants: int = DEFAULT_NUM_VARIANTS

    def __post_init__(self) -> None:
        if not (1 <= self.num_variants <= 10):
            raise ConfigurationError(
                f"num_variants must be 1-10, got {self.num_variants} — "
                "more than 10 variants dilutes recall and multiplies LLM cost"
            )

    async def rewrite(self, query: str, conversation_context: str | None = None) -> list[str]:
        registry = build_registry(self.lm_client)
        context_hint = f"\n\nConversation context: {conversation_context}" if conversation_context else ""

        try:
            result = await b.GenerateQueryVariants(
                query + context_hint,
                self.num_variants,
                baml_options={"client_registry": registry},
            )
            logger.info("multi-query: generated %d variants", len(result.variants))
            return result.variants

        except baml_errors.BamlValidationError as exc:
            logger.exception(
                "GenerateQueryVariants failed: LLM returned unparseable response — "
                "falling back to original query. Detail: %s",
                exc,
            )
            return []
        except Exception as exc:
            logger.exception("GenerateQueryVariants failed: %s — falling back to original query", exc)
            return []
