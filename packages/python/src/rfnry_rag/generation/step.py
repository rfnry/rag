from baml_py import errors as baml_errors

from rfnry_rag.baml.baml_client.async_client import b
from rfnry_rag.common.logging import get_logger
from rfnry_rag.exceptions import GenerationError
from rfnry_rag.generation.models import StepResult
from rfnry_rag.models import RetrievedChunk
from rfnry_rag.providers import LanguageModelClient, build_registry
from rfnry_rag.retrieval.common.formatting import ChunkOrdering, chunks_to_context

logger = get_logger("generation/step")


class StepGenerationService:
    """Single reasoning step generation for iterative retrieval loops."""

    def __init__(
        self,
        lm_client: LanguageModelClient,
        chunk_ordering: ChunkOrdering = ChunkOrdering.SCORE_DESCENDING,
    ) -> None:
        self._lm_client = lm_client
        self._chunk_ordering = chunk_ordering
        self._registry = build_registry(self._lm_client)

    async def generate_step(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        context: str | None = None,
    ) -> StepResult:
        """Generate a single reasoning step from retrieved chunks."""
        if not query or not query.strip():
            raise GenerationError("Query must not be empty")

        chunk_context = chunks_to_context(chunks, ordering=self._chunk_ordering) if chunks else "(No context retrieved)"
        prior_reasoning = context or ""

        try:
            result = await b.GenerateReasoningStep(
                query=query,
                context=chunk_context,
                prior_reasoning=prior_reasoning,
                baml_options={"client_registry": self._registry},
            )

            text = result.text.strip()

            logger.info(
                "reasoning step: %d chars, is_final=%s",
                len(text),
                result.is_final,
            )

            return StepResult(text=text, done=bool(result.is_final))

        except baml_errors.BamlValidationError as exc:
            raise GenerationError(f"GenerateReasoningStep returned unparseable response: {exc}") from exc
        except Exception as exc:
            raise GenerationError(f"GenerateReasoningStep failed: {exc}") from exc
