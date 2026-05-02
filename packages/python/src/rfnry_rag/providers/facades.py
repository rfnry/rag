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
from rfnry_rag.providers.provider import (
    AnthropicModelProvider,
    CohereModelProvider,
    GoogleModelProvider,
    ModelProvider,
    OpenAIModelProvider,
    VoyageModelProvider,
)
from rfnry_rag.retrieval.search.reranking.cohere import _CohereReranking
from rfnry_rag.retrieval.search.reranking.voyage import _VoyageReranking
from rfnry_rag.telemetry.usage import instrument_call


class Embeddings:
    def __init__(self, provider: ModelProvider) -> None:
        self._provider = provider
        if isinstance(provider, OpenAIModelProvider):
            self._impl: _OpenAIEmbeddings | _VoyageEmbeddings | _CohereEmbeddings = _OpenAIEmbeddings(provider)
        elif isinstance(provider, VoyageModelProvider):
            self._impl = _VoyageEmbeddings(provider)
        elif isinstance(provider, CohereModelProvider):
            self._impl = _CohereEmbeddings(provider)
        else:
            raise ConfigurationError(
                f"Unsupported embeddings provider: {provider.kind!r}. Supported: openai, voyage, cohere."
            )

    @property
    def model(self) -> str:
        return self._impl.model

    @property
    def name(self) -> str:
        return self._provider.name

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return await instrument_call(
            provider=self._provider.kind,
            model=self._provider.model,
            operation="embed",
            extract_usage=lambda _resp: {},
            call=lambda: self._impl.embed(texts),
        )

    async def embedding_dimension(self) -> int:
        return await self._impl.embedding_dimension()


class Vision:
    def __init__(
        self,
        provider: ModelProvider,
        max_tokens: int = 4096,
        max_retries: int = 3,
    ) -> None:
        self._provider = provider
        if isinstance(provider, AnthropicModelProvider):
            self._impl: _AnthropicVision | _OpenAIVision | _GeminiVision = _AnthropicVision(
                provider, max_tokens=max_tokens, max_retries=max_retries
            )
        elif isinstance(provider, OpenAIModelProvider):
            self._impl = _OpenAIVision(provider, max_tokens=max_tokens, max_retries=max_retries)
        elif isinstance(provider, GoogleModelProvider):
            self._impl = _GeminiVision(provider, max_tokens=max_tokens, max_retries=max_retries)
        else:
            raise ConfigurationError(
                f"Unsupported vision provider: {provider.kind!r}. Supported: anthropic, openai, google."
            )

    async def parse(self, file_path: str, pages: set[int] | None = None) -> list[ParsedPage]:
        return await instrument_call(
            provider=self._provider.kind,
            model=self._provider.model,
            operation="vision_parse",
            extract_usage=lambda _resp: {},
            call=lambda: self._impl.parse(file_path, pages),
        )


class Reranking:
    def __init__(self, provider: ModelProvider) -> None:
        self._provider = provider
        if isinstance(provider, CohereModelProvider):
            self._impl: _CohereReranking | _VoyageReranking = _CohereReranking(provider)
        elif isinstance(provider, VoyageModelProvider):
            self._impl = _VoyageReranking(provider)
        else:
            raise ConfigurationError(
                f"Provider {provider.kind!r} has no dedicated reranker API. Supported: cohere, voyage."
            )

    async def rerank(self, query: str, results: list[RetrievedChunk], top_k: int = 5) -> list[RetrievedChunk]:
        return await instrument_call(
            provider=self._provider.kind,
            model=self._provider.model,
            operation="rerank",
            extract_usage=lambda _resp: {},
            call=lambda: self._impl.rerank(query, results, top_k),
        )
