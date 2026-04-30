from dataclasses import replace

import cohere

from rfnry_rag.logging import get_logger
from rfnry_rag.models import RetrievedChunk
from rfnry_rag.providers.provider import LanguageModelProvider

logger = get_logger(__name__)


class _CohereReranking:
    def __init__(self, provider: LanguageModelProvider) -> None:
        self._client = cohere.AsyncClientV2(api_key=provider.api_key)
        self._model = provider.model

    async def rerank(self, query: str, results: list[RetrievedChunk], top_k: int = 5) -> list[RetrievedChunk]:
        if not results:
            return []

        documents = [r.content for r in results]

        try:
            response = await self._client.rerank(
                query=query,
                documents=documents,
                model=self._model,
                top_n=top_k,
            )
        except Exception:
            logger.exception("cohere rerank failed, returning unranked")
            return results[:top_k]

        reranked = []
        for item in response.results:
            reranked.append(replace(results[item.index], score=item.relevance_score))

        return reranked
