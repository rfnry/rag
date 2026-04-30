import pytest

from rfnry_rag.exceptions import ConfigurationError
from rfnry_rag.ingestion.embeddings.cohere import _CohereEmbeddings
from rfnry_rag.ingestion.embeddings.openai import _OpenAIEmbeddings
from rfnry_rag.ingestion.embeddings.voyage import _VoyageEmbeddings
from rfnry_rag.providers import Embeddings, LanguageModel


def test_embeddings_dispatches_to_openai():
    provider = LanguageModel(provider="openai", model="text-embedding-3-small", api_key="sk-test")
    embeddings = Embeddings(provider)
    assert isinstance(embeddings._impl, _OpenAIEmbeddings)
    assert embeddings._impl.model == "text-embedding-3-small"


def test_embeddings_dispatches_to_voyage():
    provider = LanguageModel(provider="voyage", model="voyage-3", api_key="vo-test")
    embeddings = Embeddings(provider)
    assert isinstance(embeddings._impl, _VoyageEmbeddings)


def test_embeddings_dispatches_to_cohere():
    provider = LanguageModel(provider="cohere", model="embed-english-v3.0", api_key="co-test")
    embeddings = Embeddings(provider)
    assert isinstance(embeddings._impl, _CohereEmbeddings)


def test_embeddings_unsupported_provider_raises():
    provider = LanguageModel(provider="unknown", model="m", api_key="k")
    with pytest.raises(ConfigurationError, match="Unsupported embeddings provider"):
        Embeddings(provider)


def test_embeddings_model_property_delegates():
    provider = LanguageModel(provider="openai", model="text-embedding-3-large", api_key="sk-test")
    embeddings = Embeddings(provider)
    assert embeddings.model == "text-embedding-3-large"
