from collections.abc import AsyncIterator

from rfnry_knowledge.common.logging import get_logger
from rfnry_knowledge.exceptions import GenerationError
from rfnry_knowledge.generation.formatting import ChunkOrdering, chunks_to_context
from rfnry_knowledge.generation.grounding import DEFAULT_ESCALATION, RelevanceGate, ScoreGate
from rfnry_knowledge.generation.models import Clarification, QueryResult, SourceReference, StreamEvent
from rfnry_knowledge.models import RetrievedChunk
from rfnry_knowledge.providers import LLMClient
from rfnry_knowledge.providers.text_generation import assemble_user_message

logger = get_logger("generation")


class GenerationService:
    def __init__(
        self,
        lm_client: LLMClient,
        system_prompt: str,
        grounding_enabled: bool = False,
        grounding_threshold: float = 0.5,
        relevance_gate_enabled: bool = False,
        guiding_enabled: bool = False,
        relevance_gate_lm_client: LLMClient | None = None,
        chunk_ordering: ChunkOrdering = ChunkOrdering.SCORE_DESCENDING,
    ) -> None:
        self._lm_client = lm_client
        self._system_prompt = system_prompt
        self._guiding_enabled = guiding_enabled
        self._chunk_ordering = chunk_ordering

        self._grounding_enabled = grounding_enabled
        self._score_gate = ScoreGate(threshold=grounding_threshold) if grounding_enabled else None

        self._relevance_gate: RelevanceGate | None = None
        if relevance_gate_enabled and relevance_gate_lm_client and self._score_gate:
            self._relevance_gate = RelevanceGate(lm_client=relevance_gate_lm_client, fallback_gate=self._score_gate)

    @staticmethod
    def _format_history(history: list[tuple[str, str]] | None) -> str:
        """Format conversation history as alternating role messages."""
        if not history:
            return ""
        parts = []
        for human_msg, assistant_msg in history:
            parts.append(f"User: {human_msg}")
            parts.append(f"Assistant: {assistant_msg}")
        return "\n".join(parts)

    def _run_grounding_gates(self, chunks: list[RetrievedChunk]) -> tuple[bool, str | None]:
        """Run score gate (sync). Returns (passed, escalation_message)."""
        if self._score_gate and not self._relevance_gate:
            passed, message = self._score_gate.check(chunks)
            if not passed:
                return False, message or DEFAULT_ESCALATION
        return True, None

    async def _run_relevance_gate(
        self, query: str, chunks: list[RetrievedChunk]
    ) -> tuple[bool, str | None, list[RetrievedChunk], float, QueryResult | None]:
        """Run relevance gate (async). Returns (passed, message, relevant_chunks, confidence, early_result)."""
        relevant_chunks = chunks
        confidence = 0.0
        if self._relevance_gate:
            passed, message, relevance = await self._relevance_gate.check(query, chunks)
            confidence = relevance.confidence if relevance else 0.0

            if not passed:
                if (
                    self._guiding_enabled
                    and relevance
                    and relevance.needs_clarification
                    and relevance.clarifying_question
                ):
                    logger.info("relevance gate requests clarification (guiding enabled)")
                    options = relevance.clarifying_options or []
                    if not any(opt.lower() == "something else" for opt in options):
                        options.append("Something else")
                    return (
                        False,
                        None,
                        chunks,
                        confidence,
                        QueryResult(
                            answer=None,
                            grounded=False,
                            confidence=confidence,
                            sources=[],
                            clarification=Clarification(
                                question=relevance.clarifying_question,
                                options=options,
                            ),
                        ),
                    )

                logger.info("relevance gate rejected query")
                return False, message or DEFAULT_ESCALATION, chunks, confidence, None

            if relevance and relevance.relevant_indices:
                relevant_chunks = [chunks[i] for i in relevance.relevant_indices if i < len(chunks)]
        elif self._grounding_enabled:
            confidence = max(c.score for c in chunks) if chunks else 0.0

        return True, None, relevant_chunks, confidence, None

    async def generate(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        history: list[tuple[str, str]] | None = None,
        system_prompt: str | None = None,
    ) -> QueryResult:
        """Run grounding gates then generate a response via the native LLM call."""
        if not query or not query.strip():
            raise GenerationError("Query must not be empty")

        passed, message = self._run_grounding_gates(chunks)
        if not passed:
            logger.info("score gate rejected query")
            return self._escalation_result(message or DEFAULT_ESCALATION)

        passed, message, relevant_chunks, confidence, early_result = await self._run_relevance_gate(query, chunks)
        if not passed:
            if early_result:
                return early_result
            return self._escalation_result(message or DEFAULT_ESCALATION, confidence)

        context = chunks_to_context(relevant_chunks, ordering=self._chunk_ordering)
        formatted_history = self._format_history(history)
        active_system_prompt = system_prompt if system_prompt is not None else self._system_prompt

        logger.info("LLM generation: %d context chunks", len(relevant_chunks))
        try:
            answer = await self._lm_client.generate_text(
                system_prompt=active_system_prompt,
                history=formatted_history,
                user=assemble_user_message(query=query, context=context),
            )
        except Exception as exc:
            raise GenerationError(f"LLM generation failed: {exc}") from exc
        logger.info("LLM response: %d chars", len(answer))

        sources = self._build_sources(relevant_chunks)

        return QueryResult(
            answer=answer,
            grounded=True,
            confidence=confidence,
            sources=sources,
        )

    async def generate_from_corpus(
        self,
        query: str,
        corpus: str,
        history: list[tuple[str, str]] | None = None,
        system_prompt: str | None = None,
    ) -> QueryResult:
        """Generate from a full corpus string (DIRECT mode).

        Skips grounding and clarification gates by design: with the entire
        corpus in the prompt, the chunk-level relevance signal those gates
        depend on no longer applies. Returns `sources=[]` because DIRECT
        does not assemble chunk-level source attribution.
        """
        if not query or not query.strip():
            raise GenerationError("Query must not be empty")

        formatted_history = self._format_history(history)
        active_system_prompt = system_prompt if system_prompt is not None else self._system_prompt

        logger.info("LLM generation (direct corpus): %d chars", len(corpus))
        try:
            answer = await self._lm_client.generate_text(
                system_prompt=active_system_prompt,
                history=formatted_history,
                user=assemble_user_message(query=query, context=corpus),
            )
        except Exception as exc:
            raise GenerationError(f"LLM generation failed: {exc}") from exc
        logger.info("LLM response: %d chars", len(answer))

        return QueryResult(
            answer=answer,
            grounded=True,
            confidence=0.0,
            sources=[],
        )

    async def generate_stream(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        history: list[tuple[str, str]] | None = None,
        system_prompt: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Run grounding gates then stream a response via the native LLM call."""
        if not query or not query.strip():
            raise GenerationError("Query must not be empty")

        passed, message = self._run_grounding_gates(chunks)
        if not passed:
            logger.info("score gate rejected query (stream)")
            yield StreamEvent(type="done", content=message or DEFAULT_ESCALATION, grounded=False)
            return

        passed, message, relevant_chunks, confidence, early_result = await self._run_relevance_gate(query, chunks)
        if not passed:
            if early_result:
                yield StreamEvent(
                    type="done",
                    grounded=False,
                    confidence=confidence,
                    clarification=early_result.clarification,
                )
            else:
                yield StreamEvent(type="done", content=message or DEFAULT_ESCALATION, grounded=False)
            return

        context = chunks_to_context(relevant_chunks, ordering=self._chunk_ordering)
        formatted_history = self._format_history(history)
        active_system_prompt = system_prompt if system_prompt is not None else self._system_prompt

        logger.info("LLM streaming: %d context chunks", len(relevant_chunks))
        try:
            stream = self._lm_client.generate_text_stream(
                system_prompt=active_system_prompt,
                history=formatted_history,
                user=assemble_user_message(query=query, context=context),
            )
            async for delta in stream:
                if delta:
                    yield StreamEvent(type="chunk", content=delta)
        except Exception as exc:
            # Emit a terminal event before raising so consumers iterating the
            # async generator always see a structured error marker, not just
            # an unexpected exception out of the for-loop.
            yield StreamEvent(type="done", content=f"generation error: {exc}", grounded=False)
            raise GenerationError(f"LLM streaming failed: {exc}") from exc

        sources = self._build_sources(relevant_chunks)
        yield StreamEvent(type="sources", sources=sources, grounded=True, confidence=confidence)

    @staticmethod
    def _build_sources(chunks: list[RetrievedChunk]) -> list[SourceReference]:
        return [
            SourceReference(
                source_id=c.source_id,
                name=c.source_metadata.get("name", ""),
                page_number=c.page_number,
                section=c.section,
                score=c.score,
                file_url=c.source_metadata.get("file_url"),
            )
            for c in chunks
        ]

    @staticmethod
    def _escalation_result(
        message: str,
        confidence: float = 0.0,
    ) -> QueryResult:
        return QueryResult(
            answer=message,
            grounded=False,
            confidence=confidence,
            sources=[],
        )
