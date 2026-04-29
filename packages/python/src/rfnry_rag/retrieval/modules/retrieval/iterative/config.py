"""IterativeRetrievalConfig — multi-hop iterative retrieval configuration (R6.1).

The hop loop, engine integration, and trace surface land in R6.2; post-loop
DIRECT escalation lands in R6.3. R6.1 is plumbing only — `enabled=False` is
the default and no consumer reads these fields yet.
"""

from __future__ import annotations

from dataclasses import dataclass

from rfnry_rag.retrieval.common.errors import ConfigurationError
from rfnry_rag.retrieval.common.language_model import LanguageModelClient

_VALID_GATE_MODES = frozenset({"type", "llm"})


@dataclass
class IterativeRetrievalConfig:
    """Multi-hop iterative retrieval configuration.

    Default-off. When ``enabled=True`` (R6.2), multi-hop queries identified by
    R5's classifier are decomposed into sequential sub-questions, each
    retrieved independently. Findings accumulate across hops via the
    decomposer's self-summarisation; raw chunks deduplicated by ``chunk_id``
    are passed to final generation unchanged.

    ``gate_mode`` selects how the loop decides whether to continue:
      - ``"type"``: gate on R5 classifier's ``QueryType`` (cheap, deterministic).
      - ``"llm"``: gate via the ``DecomposeQuery`` BAML function (slower, more
        accurate on ambiguous queries; requires ``decomposition_model``).

    ``grounding_threshold=None`` (default) inherits ``GenerationConfig.
    grounding_threshold`` at the consume site — keeping a single source of
    truth when the consumer doesn't need a per-mode override. Setting an
    explicit value here decouples iterative-mode confidence from the global
    knob.
    """

    enabled: bool = False
    max_hops: int = 3
    decomposition_model: LanguageModelClient | None = None
    gate_mode: str = "type"
    escalate_to_direct: bool = True
    grounding_threshold: float | None = None

    def __post_init__(self) -> None:
        if not (1 <= self.max_hops <= 10):
            raise ConfigurationError(
                f"IterativeRetrievalConfig.max_hops must be in [1, 10], got {self.max_hops}"
            )
        if self.gate_mode not in _VALID_GATE_MODES:
            raise ConfigurationError(
                f"IterativeRetrievalConfig.gate_mode must be one of {sorted(_VALID_GATE_MODES)}, got {self.gate_mode!r}"
            )
        if self.grounding_threshold is not None and not (0.0 <= self.grounding_threshold <= 1.0):
            raise ConfigurationError(
                f"IterativeRetrievalConfig.grounding_threshold must be in [0.0, 1.0] when set, "
                f"got {self.grounding_threshold}"
            )
        # LLM-gated decomposition requires a model: rejecting at construction
        # surfaces the misconfig before the first hop runs (vs. blowing up
        # mid-loop on the first BAML call with a stale stack).
        if self.enabled and self.gate_mode == "llm" and self.decomposition_model is None:
            raise ConfigurationError("IterativeRetrievalConfig.gate_mode='llm' requires decomposition_model")
