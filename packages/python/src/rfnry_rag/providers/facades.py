from __future__ import annotations

from rfnry_rag.exceptions import ConfigurationError
from rfnry_rag.ingestion.embeddings.cohere import _CohereEmbeddings
from rfnry_rag.ingestion.embeddings.openai import _OpenAIEmbeddings
from rfnry_rag.ingestion.embeddings.voyage import _VoyageEmbeddings
from rfnry_rag.ingestion.models import ParsedPage
from rfnry_rag.ingestion.vision.anthropic import _AnthropicVision
from rfnry_rag.ingestion.vision.gemini import _GeminiVision
from rfnry_rag.ingestion.vision.openai import _OpenAIVision
from rfnry_rag.models import RetrievedChunk
from rfnry_rag.providers.provider import LanguageModelProvider
from rfnry_rag.retrieval.search.reranking.cohere import _CohereReranking
from rfnry_rag.retrieval.search.reranking.voyage import _VoyageReranking

_DEDICATED_RERANKER_PROVIDERS = {"cohere", "voyage"}


class Embeddings:
    """Embeddings client dispatching to the correct provider implementation."""

    def __init__(self, provider: LanguageModelProvider) -> None:
        self._provider = provider
        match provider.provider:
            case "openai":
                self._impl: _OpenAIEmbeddings | _VoyageEmbeddings | _CohereEmbeddings = _OpenAIEmbeddings(provider)
            case "voyage":
                self._impl = _VoyageEmbeddings(provider)
            case "cohere":
                self._impl = _CohereEmbeddings(provider)
            case _:
                raise ConfigurationError(
                    f"Unsupported embeddings provider: {provider.provider!r}. Supported: openai, voyage, cohere."
                )

    @property
    def model(self) -> str:
        return self._impl.model

    @property
    def name(self) -> str:
        return self._provider.name

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return await self._impl.embed(texts)

    async def embedding_dimension(self) -> int:
        return await self._impl.embedding_dimension()


class Vision:
    """Vision client dispatching to the correct provider implementation."""

    def __init__(
        self,
        provider: LanguageModelProvider,
        max_tokens: int = 4096,
        max_retries: int = 3,
    ) -> None:
        match provider.provider:
            case "anthropic":
                self._impl: _AnthropicVision | _OpenAIVision | _GeminiVision = _AnthropicVision(
                    provider, max_tokens=max_tokens, max_retries=max_retries
                )
            case "openai":
                self._impl = _OpenAIVision(provider, max_tokens=max_tokens, max_retries=max_retries)
            case "gemini":
                self._impl = _GeminiVision(provider, max_tokens=max_tokens, max_retries=max_retries)
            case _:
                raise ConfigurationError(
                    f"Unsupported vision provider: {provider.provider!r}. Supported: anthropic, openai, gemini."
                )

    async def parse(self, file_path: str, pages: set[int] | None = None) -> list[ParsedPage]:
        return await self._impl.parse(file_path, pages)


class Reranking:
    """Reranker facade dispatching to a dedicated provider API (Cohere or Voyage)."""

    def __init__(self, provider: LanguageModelProvider) -> None:
        if provider.provider not in _DEDICATED_RERANKER_PROVIDERS:
            raise ConfigurationError(
                f"Provider {provider.provider!r} has no dedicated reranker API. "
                f"Supported: {', '.join(sorted(_DEDICATED_RERANKER_PROVIDERS))}."
            )
        self._impl: _CohereReranking | _VoyageReranking = (
            _CohereReranking(provider) if provider.provider == "cohere" else _VoyageReranking(provider)
        )

    async def rerank(self, query: str, results: list[RetrievedChunk], top_k: int = 5) -> list[RetrievedChunk]:
        return await self._impl.rerank(query, results, top_k)
