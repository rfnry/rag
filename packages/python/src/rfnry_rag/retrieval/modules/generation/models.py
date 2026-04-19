from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class SourceReference:
    source_id: str
    name: str = ""
    page_number: int | None = None
    section: str | None = None
    score: float = 0.0
    file_url: str | None = None


@dataclass
class Clarification:
    question: str
    options: list[str] = field(default_factory=list)


@dataclass
class QueryResult:
    answer: str | None
    sources: list[SourceReference]
    grounded: bool = False
    confidence: float = 0.0
    clarification: Clarification | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamEvent:
    type: Literal["chunk", "sources", "done"]
    content: str | None = None
    sources: list[SourceReference] | None = None
    grounded: bool = False
    confidence: float = 0.0
    clarification: Clarification | None = None


@dataclass
class RelevanceResult:
    answerable: bool
    confidence: float
    relevant_indices: list[int]
    needs_clarification: bool = False
    clarifying_question: str | None = None
    clarifying_options: list[str] | None = field(default=None)


@dataclass
class StepResult:
    """Result from a single reasoning step generation."""

    text: str
    done: bool
