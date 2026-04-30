import pytest

from rfnry_rag.common.errors import ConfigurationError
from rfnry_rag.common.language_model import LanguageModelClient, LanguageModelProvider
from rfnry_rag.retrieval.search.reranking.cohere import _CohereReranking
from rfnry_rag.retrieval.search.reranking.facade import Reranking
from rfnry_rag.retrieval.search.reranking.llm import _LLMReranking
from rfnry_rag.retrieval.search.reranking.voyage import _VoyageReranking


def test_reranking_with_cohere_provider_uses_dedicated_api():
    provider = LanguageModelProvider(provider="cohere", model="rerank-v3.5", api_key="co-test")
    reranker = Reranking(provider)
    assert isinstance(reranker._impl, _CohereReranking)


def test_reranking_with_voyage_provider_uses_dedicated_api():
    provider = LanguageModelProvider(provider="voyage", model="rerank-2.5", api_key="vo-test")
    reranker = Reranking(provider)
    assert isinstance(reranker._impl, _VoyageReranking)


def test_reranking_with_client_uses_llm_path():
    client = LanguageModelClient(
        provider=LanguageModelProvider(provider="anthropic", model="claude-sonnet-4-20250514", api_key="sk-test"),
    )
    reranker = Reranking(client)
    assert isinstance(reranker._impl, _LLMReranking)


def test_reranking_with_unsupported_provider_raises():
    provider = LanguageModelProvider(provider="openai", model="gpt-4o", api_key="sk-test")
    with pytest.raises(ConfigurationError, match="no dedicated reranker API"):
        Reranking(provider)


def test_voyage_reranker_uses_async_client() -> None:
    import voyageai

    from rfnry_rag.retrieval.search.reranking.voyage import _VoyageReranking

    rerank = _VoyageReranking(provider=LanguageModelProvider(provider="voyage", model="rerank-2", api_key="sk-test"))
    assert isinstance(rerank._client, voyageai.AsyncClient)
