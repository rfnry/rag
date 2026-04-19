from __future__ import annotations

from rfnry_rag.reasoning.baml.baml_client.async_client import b
from rfnry_rag.reasoning.common.concurrency import run_concurrent
from rfnry_rag.reasoning.common.errors import AnalysisError
from rfnry_rag.reasoning.common.language_model import LanguageModelClient, build_registry
from rfnry_rag.reasoning.common.logging import get_logger
from rfnry_rag.reasoning.modules.analysis.models import (
    AnalysisConfig,
    AnalysisResult,
    DimensionResult,
    Entity,
    IntentShift,
    Message,
    RetrievalHint,
)

logger = get_logger("analysis")


def _build_instructions(cfg: AnalysisConfig) -> str:
    """Build the instruction string for the BAML prompt from config."""
    parts = ["Identify the primary intent of this text with a confidence score (0.0-1.0)."]

    if cfg.dimensions:
        dim_text = "\n".join(f"- **{d.name}**: {d.description} (scale: {d.scale})" for d in cfg.dimensions)
        parts.append(f"Score the text along these dimensions:\n{dim_text}")

    if cfg.entity_types:
        ent_text = "\n".join(f"- **{e.name}**: {e.description}" for e in cfg.entity_types)
        parts.append(f"Extract entities of these types (return empty list if none found):\n{ent_text}")

    if cfg.generate_retrieval_hints and cfg.retrieval_hint_scopes:
        scopes = ", ".join(cfg.retrieval_hint_scopes)
        parts.append(
            f"Suggest retrieval queries for these knowledge scopes: {scopes}\n"
            "For each hint, specify which scope it targets, a search query, "
            "why this retrieval would help, and priority (0.0-1.0)."
        )

    if cfg.summarize:
        parts.append("Provide a 1-2 sentence summary of the text.")

    return "\n\n".join(parts)


def _build_context_instructions(cfg: AnalysisConfig) -> str:
    """Build instructions for thread analysis, including tracking directives."""
    parts = [_build_instructions(cfg)]

    if cfg.context_tracking:
        tc = cfg.context_tracking
        thread_parts = []
        if tc.track_intent_shifts:
            thread_parts.append(
                "- Detect any intent shifts during the conversation (what changed, at which message index, and why)"
            )
        if tc.detect_escalation:
            thread_parts.append("- Detect whether escalation signals are present (and explain why)")
        if tc.track_resolution:
            thread_parts.append("- Determine the resolution status: resolved, pending, or escalated")
        if thread_parts:
            parts.append("Thread-level analysis:\n" + "\n".join(thread_parts))

    return "\n\n".join(parts)


def _parse_dimensions(raw: list) -> dict[str, DimensionResult]:
    return {
        d.name: DimensionResult(
            name=d.name,
            value=d.value,
            confidence=float(d.confidence),
            reasoning=d.reasoning,
        )
        for d in raw
    }


def _parse_entities(raw: list) -> list[Entity]:
    return [Entity(type=e.type, value=e.value, context=e.context) for e in raw]


def _parse_hints(raw: list) -> list[RetrievalHint]:
    return [
        RetrievalHint(
            query=h.query,
            knowledge_scope=h.knowledge_scope,
            reasoning=h.reasoning,
            priority=float(h.priority),
        )
        for h in raw
    ]


class AnalysisService:
    """Extract structured insights from text or conversation threads."""

    def __init__(self, lm_client: LanguageModelClient) -> None:
        self._registry = build_registry(lm_client)

    async def analyze(
        self,
        text: str,
        config: AnalysisConfig | None = None,
    ) -> AnalysisResult:
        """Analyze a single text input."""
        cfg = config or AnalysisConfig()
        truncated = text[: cfg.max_text_length]
        instructions = _build_instructions(cfg)
        logger.info("[analysis/text] analyzing text (%d chars)", len(truncated))

        try:
            result = await b.AnalyzeText(
                truncated,
                instructions,
                baml_options={"client_registry": self._registry},
            )
            logger.info("[analysis/text] intent: '%s' (confidence: %.2f)", result.primary_intent, result.confidence)
            return AnalysisResult(
                primary_intent=result.primary_intent,
                confidence=float(result.confidence),
                dimensions=_parse_dimensions(result.dimensions or []),
                entities=_parse_entities(result.entities or []),
                summary=result.summary,
                retrieval_hints=_parse_hints(result.retrieval_hints or []),
            )
        except AnalysisError:
            raise
        except Exception as exc:
            raise AnalysisError(f"Analysis failed: {exc}") from exc

    async def analyze_context(
        self,
        messages: list[Message],
        config: AnalysisConfig | None = None,
    ) -> AnalysisResult:
        """Analyze a multi-turn thread with optional thread tracking."""
        cfg = config or AnalysisConfig()
        formatted_messages = "\n".join(f"[{i}] {m.role}: {m.text}" for i, m in enumerate(messages))
        roles = ", ".join(sorted({m.role for m in messages}))
        instructions = _build_context_instructions(cfg)
        logger.info("[analysis/context] analyzing %d messages", len(messages))

        try:
            result = await b.AnalyzeContext(
                formatted_messages,
                roles,
                instructions,
                baml_options={"client_registry": self._registry},
            )
            intent_shifts = [
                IntentShift(
                    from_intent=s.from_intent,
                    to_intent=s.to_intent,
                    at_message=s.at_message,
                    reasoning=s.reasoning,
                )
                for s in (result.intent_shifts or [])
            ]
            logger.info(
                "[analysis/context] intent: '%s', escalation: %s, status: %s",
                result.primary_intent,
                result.escalation_detected,
                result.resolution_status,
            )
            return AnalysisResult(
                primary_intent=result.primary_intent,
                confidence=float(result.confidence),
                dimensions=_parse_dimensions(result.dimensions or []),
                entities=_parse_entities(result.entities or []),
                summary=result.summary,
                retrieval_hints=_parse_hints(result.retrieval_hints or []),
                intent_shifts=intent_shifts,
                escalation_detected=result.escalation_detected,
                escalation_reasoning=result.escalation_reasoning,
                resolution_status=result.resolution_status,
            )
        except AnalysisError:
            raise
        except Exception as exc:
            raise AnalysisError(f"Thread analysis failed: {exc}") from exc

    async def analyze_batch(
        self,
        texts: list[str],
        config: AnalysisConfig | None = None,
    ) -> list[AnalysisResult]:
        """Analyze multiple texts concurrently."""
        cfg = config or AnalysisConfig()

        async def _analyze_one(text: str) -> AnalysisResult:
            return await self.analyze(text, cfg)

        return await run_concurrent(texts, _analyze_one, cfg.concurrency)
