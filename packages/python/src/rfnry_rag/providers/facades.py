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
from rfnry_rag.providers.provider import LanguageModel
from rfnry_rag.retrieval.search.reranking.cohere import _CohereReranking
from rfnry_rag.retrieval.search.reranking.voyage import _VoyageReranking
from rfnry_rag.telemetry.usage import instrument_call

_DEDICATED_RERANKER_PROVIDERS = {"cohere", "voyage"}


class Embeddings:
    """Embeddings client dispatching to the correct provider implementation."""

    def __init__(self, lm: LanguageModel) -> None:
        self._lm = lm
        match lm.provider:
            case "openai":
                self._impl: _OpenAIEmbeddings | _VoyageEmbeddings | _CohereEmbeddings = _OpenAIEmbeddings(lm)
            case "voyage":
                self._impl = _VoyageEmbeddings(lm)
            case "cohere":
                self._impl = _CohereEmbeddings(lm)
            case _:
                raise ConfigurationError(
                    f"Unsupported embeddings provider: {lm.provider!r}. Supported: openai, voyage, cohere."
                )

    @property
    def model(self) -> str:
        return self._impl.model

    @property
    def name(self) -> str:
        return self._lm.name

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return await instrument_call(
            provider=self._lm.provider,
            model=self._lm.model,
            operation="embed",
            extract_usage=lambda _resp: {},
            call=lambda: self._impl.embed(texts),
        )

    async def embedding_dimension(self) -> int:
        return await self._impl.embedding_dimension()


class Vision:
    """Vision client dispatching to the correct provider implementation."""

    def __init__(
        self,
        lm: LanguageModel,
        max_tokens: int = 4096,
        max_retries: int = 3,
    ) -> None:
        self._lm = lm
        match lm.provider:
            case "anthropic":
                self._impl: _AnthropicVision | _OpenAIVision | _GeminiVision = _AnthropicVision(
                    lm, max_tokens=max_tokens, max_retries=max_retries
                )
            case "openai":
                self._impl = _OpenAIVision(lm, max_tokens=max_tokens, max_retries=max_retries)
            case "gemini":
                self._impl = _GeminiVision(lm, max_tokens=max_tokens, max_retries=max_retries)
            case _:
                raise ConfigurationError(
                    f"Unsupported vision provider: {lm.provider!r}. Supported: anthropic, openai, gemini."
                )

    async def parse(self, file_path: str, pages: set[int] | None = None) -> list[ParsedPage]:
        return await instrument_call(
            provider=self._lm.provider,
            model=self._lm.model,
            operation="vision_parse",
            extract_usage=lambda _resp: {},
            call=lambda: self._impl.parse(file_path, pages),
        )


class Reranking:
    """Reranker facade dispatching to a dedicated provider API (Cohere or Voyage)."""

    def __init__(self, lm: LanguageModel) -> None:
        if lm.provider not in _DEDICATED_RERANKER_PROVIDERS:
            raise ConfigurationError(
                f"Provider {lm.provider!r} has no dedicated reranker API. "
                f"Supported: {', '.join(sorted(_DEDICATED_RERANKER_PROVIDERS))}."
            )
        self._lm = lm
        self._impl: _CohereReranking | _VoyageReranking = (
            _CohereReranking(lm) if lm.provider == "cohere" else _VoyageReranking(lm)
        )

    async def rerank(self, query: str, results: list[RetrievedChunk], top_k: int = 5) -> list[RetrievedChunk]:
        return await instrument_call(
            provider=self._lm.provider,
            model=self._lm.model,
            operation="rerank",
            extract_usage=lambda _resp: {},
            call=lambda: self._impl.rerank(query, results, top_k),
        )
