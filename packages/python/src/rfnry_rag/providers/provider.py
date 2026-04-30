from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LanguageModelProvider:
    """Identity: which API, which model, auth key.

    Used directly by Embeddings/Vision/Reranking for dedicated provider APIs,
    or wrapped by LanguageModelClient for BAML-routed LLM calls.

    ``name`` is an optional stable identifier used as a fingerprint for
    cross-process consistency checks (e.g. embedding-model migration). When
    omitted it defaults to ``"{provider}:{model}"``.
    """

    provider: str
    model: str
    api_key: str | None = field(default=None, repr=False)
    name: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            self.name = f"{self.provider}:{self.model}"
