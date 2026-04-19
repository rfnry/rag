from __future__ import annotations

from rfnry_rag.common.errors import ConfigurationError
from rfnry_rag.common.language_model import LanguageModelClient, LanguageModelProvider
from rfnry_rag.retrieval.common.models import RetrievedChunk
from rfnry_rag.retrieval.modules.retrieval.search.reranking.cohere import _CohereReranking
from rfnry_rag.retrieval.modules.retrieval.search.reranking.llm import _LLMReranking
from rfnry_rag.retrieval.modules.retrieval.search.reranking.voyage import _VoyageReranking

_DEDICATED_RERANKER_PROVIDERS = {"cohere", "voyage"}


class Reranking:
    """Unified reranker facade.

    Accepts either a LanguageModelProvider (dispatches to a dedicated reranker API —
    Cohere or Voyage) or a LanguageModelClient (dispatches to LLM-as-reranker via BAML).
    """

    def __init__(self, config: LanguageModelProvider | LanguageModelClient) -> None:
        if isinstance(config, LanguageModelClient):
            self._impl: _LLMReranking | _CohereReranking | _VoyageReranking = _LLMReranking(config)
        else:
            if config.provider not in _DEDICATED_RERANKER_PROVIDERS:
                raise ConfigurationError(
                    f"Provider {config.provider!r} has no dedicated reranker API. "
                    f"Wrap it in LanguageModelClient to use LLM-as-reranker."
                )
            self._impl = _CohereReranking(config) if config.provider == "cohere" else _VoyageReranking(config)

    async def rerank(self, query: str, results: list[RetrievedChunk], top_k: int = 5) -> list[RetrievedChunk]:
        return await self._impl.rerank(query, results, top_k)
