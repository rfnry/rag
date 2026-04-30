from __future__ import annotations

from rfnry_rag.common.language_model import LanguageModelProvider
from rfnry_rag.exceptions import ConfigurationError
from rfnry_rag.ingestion.embeddings.cohere import _CohereEmbeddings
from rfnry_rag.ingestion.embeddings.openai import _OpenAIEmbeddings
from rfnry_rag.ingestion.embeddings.voyage import _VoyageEmbeddings


class Embeddings:
    """Embeddings client dispatching to the correct provider implementation."""

    def __init__(self, provider: LanguageModelProvider) -> None:
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

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return await self._impl.embed(texts)

    async def embedding_dimension(self) -> int:
        return await self._impl.embedding_dimension()
