from __future__ import annotations

from dataclasses import dataclass, field

from rfnry_rag.reasoning.common.errors import ReasoningInputError


@dataclass
class ComplianceDimensionDefinition:
    """Consumer defines compliance checking axes."""

    name: str
    description: str


@dataclass
class Violation:
    """A specific breach found during compliance checking."""

    dimension: str
    description: str
    severity: str
    suggestion: str | None = None


_MAX_TEXT_LENGTH_CEILING = 5_000_000


@dataclass
class ComplianceConfig:
    """Configuration for compliance operations."""

    dimensions: list[ComplianceDimensionDefinition] | None = None
    threshold: float | None = None
    max_text_length: int = 3000
    max_reference_length: int = 5000
    concurrency: int = 10

    def __post_init__(self) -> None:
        if self.concurrency < 1:
            raise ReasoningInputError("concurrency must be >= 1")
        if self.concurrency > 20:
            raise ReasoningInputError("concurrency must be <= 20 — higher values risk overwhelming the LLM provider")
        if self.max_text_length < 1:
            raise ReasoningInputError("max_text_length must be >= 1")
        if self.max_text_length > _MAX_TEXT_LENGTH_CEILING:
            raise ReasoningInputError(
                f"max_text_length must be <= {_MAX_TEXT_LENGTH_CEILING}, got {self.max_text_length}"
            )
        if self.max_reference_length < 1:
            raise ReasoningInputError("max_reference_length must be >= 1")
        if self.max_reference_length > _MAX_TEXT_LENGTH_CEILING:
            raise ReasoningInputError(
                f"max_reference_length must be <= {_MAX_TEXT_LENGTH_CEILING}, got {self.max_reference_length}"
            )
        if self.threshold is not None and not 0.0 <= self.threshold <= 1.0:
            raise ReasoningInputError("threshold must be between 0.0 and 1.0")


@dataclass
class ComplianceResult:
    """Result of checking text compliance against a reference."""

    compliant: bool
    score: float
    violations: list[Violation]
    reasoning: str
    dimension_scores: dict[str, float] = field(default_factory=dict)
