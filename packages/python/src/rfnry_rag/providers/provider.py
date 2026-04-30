from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LanguageModelProvider:
    """Identity: which API, which model, auth key.

    Used directly by Embeddings/Vision/Reranking for dedicated provider APIs,
    or wrapped by LanguageModelClient for BAML-routed LLM calls.
    """

    provider: str
    model: str
    api_key: str | None = field(default=None, repr=False)
