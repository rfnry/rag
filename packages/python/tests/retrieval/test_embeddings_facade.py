import pytest

from rfnry_knowledge.exceptions import ConfigurationError
from rfnry_knowledge.ingestion.embeddings.cohere import _CohereEmbeddings
from rfnry_knowledge.ingestion.embeddings.openai import _OpenAIEmbeddings
from rfnry_knowledge.ingestion.embeddings.voyage import _VoyageEmbeddings
from rfnry_knowledge.providers import (
    AnthropicModelProvider,
    CohereModelProvider,
    Embeddings,
    OpenAIModelProvider,
    VoyageModelProvider,
)


def test_embeddings_dispatches_to_openai():
    embeddings = Embeddings(OpenAIModelProvider(api_key="sk-test", model="text-embedding-3-small"))
    assert isinstance(embeddings._impl, _OpenAIEmbeddings)
    assert embeddings._impl.model == "text-embedding-3-small"


def test_embeddings_dispatches_to_voyage():
    embeddings = Embeddings(VoyageModelProvider(api_key="vo-test", model="voyage-3"))
    assert isinstance(embeddings._impl, _VoyageEmbeddings)


def test_embeddings_dispatches_to_cohere():
    embeddings = Embeddings(CohereModelProvider(api_key="co-test", model="embed-english-v3.0"))
    assert isinstance(embeddings._impl, _CohereEmbeddings)


def test_embeddings_unsupported_provider_raises():
    with pytest.raises(ConfigurationError, match="Unsupported embeddings provider"):
        Embeddings(AnthropicModelProvider(api_key="k", model="m"))


def test_embeddings_model_property_delegates():
    embeddings = Embeddings(OpenAIModelProvider(api_key="sk-test", model="text-embedding-3-large"))
    assert embeddings.model == "text-embedding-3-large"
