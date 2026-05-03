import pytest

from rfnry_knowledge.exceptions import ConfigurationError
from rfnry_knowledge.providers import (
    CohereModelProvider,
    OpenAIModelProvider,
    Reranking,
    VoyageModelProvider,
)
from rfnry_knowledge.retrieval.search.reranking.cohere import _CohereReranking
from rfnry_knowledge.retrieval.search.reranking.voyage import _VoyageReranking


def test_reranking_with_cohere_provider_uses_dedicated_api():
    reranker = Reranking(CohereModelProvider(api_key="co-test", model="rerank-v3.5"))
    assert isinstance(reranker._impl, _CohereReranking)


def test_reranking_with_voyage_provider_uses_dedicated_api():
    reranker = Reranking(VoyageModelProvider(api_key="vo-test", model="rerank-2.5"))
    assert isinstance(reranker._impl, _VoyageReranking)


def test_reranking_with_unsupported_provider_raises():
    with pytest.raises(ConfigurationError, match="no dedicated reranker API"):
        Reranking(OpenAIModelProvider(api_key="sk-test", model="gpt-4o"))


def test_voyage_reranker_uses_async_client() -> None:
    import voyageai

    from rfnry_knowledge.retrieval.search.reranking.voyage import _VoyageReranking

    rerank = _VoyageReranking(provider=VoyageModelProvider(api_key="sk-test", model="rerank-2"))
    assert isinstance(rerank._client, voyageai.AsyncClient)
