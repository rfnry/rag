from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Message:
    """A single message in a conversation thread."""

    text: str
    role: str
    timestamp: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class DimensionDefinition:
    """Consumer defines what signals to score."""

    name: str
    description: str
    scale: str


@dataclass
class EntityTypeDefinition:
    """Consumer defines what entities to extract."""

    name: str
    description: str


@dataclass
class DimensionResult:
    """Scored result for a single dimension."""

    name: str
    value: str | float
    confidence: float
    reasoning: str


@dataclass
class Entity:
    """A single extracted entity."""

    type: str
    value: str
    context: str


@dataclass
class RetrievalHint:
    """Suggested retrieval query for a knowledge scope."""

    query: str
    knowledge_scope: str | None
    reasoning: str
    priority: float


@dataclass
class IntentShift:
    """A detected change in conversation intent."""

    from_intent: str
    to_intent: str
    at_message: int
    reasoning: str


@dataclass
class ContextTrackingConfig:
    """Opt-in multi-turn features for analyze_context()."""

    track_intent_shifts: bool = True
    track_resolution: bool = True
    detect_escalation: bool = True


@dataclass
class AnalysisConfig:
    """Configuration for analysis operations."""

    dimensions: list[DimensionDefinition] | None = None
    entity_types: list[EntityTypeDefinition] | None = None
    summarize: bool = False
    generate_retrieval_hints: bool = False
    retrieval_hint_scopes: list[str] | None = None
    context_tracking: ContextTrackingConfig | None = None
    max_text_length: int = 3000
    concurrency: int = 10

    def __post_init__(self) -> None:
        if self.max_text_length < 1:
            raise ValueError("max_text_length must be >= 1")
        if self.concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        if self.generate_retrieval_hints and not self.retrieval_hint_scopes:
            raise ValueError("retrieval_hint_scopes required when generate_retrieval_hints is True")


@dataclass
class AnalysisResult:
    """Result of analyzing text or a conversation thread."""

    primary_intent: str
    confidence: float
    dimensions: dict[str, DimensionResult] = field(default_factory=dict)
    entities: list[Entity] = field(default_factory=list)
    summary: str | None = None
    retrieval_hints: list[RetrievalHint] = field(default_factory=list)
    intent_shifts: list[IntentShift] = field(default_factory=list)
    escalation_detected: bool | None = None
    escalation_reasoning: str | None = None
    resolution_status: str | None = None
