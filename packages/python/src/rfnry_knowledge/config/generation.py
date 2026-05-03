from __future__ import annotations

from dataclasses import dataclass

from rfnry_knowledge.exceptions import ConfigurationError
from rfnry_knowledge.generation.formatting import ChunkOrdering
from rfnry_knowledge.providers import LLMClient

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Use only the provided context to answer questions. "
    "Cite sources with page numbers when available. If the context does not contain "
    "enough information to answer, say so."
)


@dataclass
class GenerationConfig:
    lm_client: LLMClient | None = None
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    grounding_enabled: bool = False
    grounding_threshold: float = 0.5
    relevance_gate_enabled: bool = False
    relevance_gate_model: LLMClient | None = None
    guiding_enabled: bool = False
    chunk_ordering: ChunkOrdering = ChunkOrdering.SCORE_DESCENDING

    def __post_init__(self) -> None:
        if self.grounding_threshold < 0 or self.grounding_threshold > 1:
            raise ConfigurationError("grounding_threshold must be between 0 and 1")
        if self.relevance_gate_enabled and not self.grounding_enabled:
            raise ConfigurationError("relevance_gate_enabled requires grounding_enabled")
        if self.relevance_gate_enabled and not self.relevance_gate_model:
            raise ConfigurationError("relevance_gate_enabled requires relevance_gate_model")
        if self.guiding_enabled and not self.relevance_gate_enabled:
            raise ConfigurationError("guiding_enabled requires relevance_gate_enabled")
        if self.grounding_enabled and self.grounding_threshold == 0.0:
            raise ConfigurationError("grounding_enabled=True with grounding_threshold=0.0 is a no-op")
        if self.grounding_enabled and self.lm_client is None:
            raise ConfigurationError("grounding_enabled requires lm_client")
