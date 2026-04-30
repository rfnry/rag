from __future__ import annotations

from dataclasses import dataclass, field

from rfnry_rag.exceptions import ConfigurationError


@dataclass
class LanguageModelProvider:
    """Identity: which API backend, which model, auth key, advertised window.

    Used directly by Embeddings/Vision/Reranking for dedicated provider APIs,
    or wrapped by LanguageModelClient for BAML-routed LLM calls.

    ``backend`` is the dispatch key — the SDK/API to call (e.g. ``"anthropic"``,
    ``"openai"``, ``"gemini"``, ``"voyage"``, ``"cohere"``).

    ``name`` is an optional stable identifier used as a fingerprint for
    cross-process consistency checks (e.g. embedding-model migration). When
    omitted it defaults to ``"{backend}:{model}"``.

    ``context_size`` is the model's advertised input window in tokens. It is
    a *safety cap*, not a routing threshold: when set, RagEngine init refuses
    configurations where ``RoutingConfig.full_context_threshold`` plus a
    reserve for the system prompt + history + question + max output tokens
    would exceed it. Effective context (Lost-in-the-Middle, LaRA) is
    typically lower than advertised; do not raise the routing threshold to
    match the window. ``None`` disables the cap.
    """

    backend: str
    model: str
    api_key: str | None = field(default=None, repr=False)
    name: str = ""
    context_size: int | None = None

    def __post_init__(self) -> None:
        if not self.name:
            self.name = f"{self.backend}:{self.model}"
        if self.context_size is not None and self.context_size < 1:
            raise ConfigurationError(
                f"LanguageModelProvider.context_size={self.context_size} must be a positive integer or None"
            )
