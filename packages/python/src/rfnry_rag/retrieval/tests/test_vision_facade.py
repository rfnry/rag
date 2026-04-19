import pytest

from rfnry_rag.common.errors import ConfigurationError
from rfnry_rag.common.language_model import LanguageModelProvider
from rfnry_rag.retrieval.modules.ingestion.vision.anthropic import _AnthropicVision
from rfnry_rag.retrieval.modules.ingestion.vision.facade import Vision
from rfnry_rag.retrieval.modules.ingestion.vision.openai import _OpenAIVision


def test_vision_dispatches_to_anthropic():
    provider = LanguageModelProvider(provider="anthropic", model="claude-sonnet-4-20250514", api_key="sk-test")
    vision = Vision(provider)
    assert isinstance(vision._impl, _AnthropicVision)


def test_vision_dispatches_to_openai():
    provider = LanguageModelProvider(provider="openai", model="gpt-4o", api_key="sk-test")
    vision = Vision(provider)
    assert isinstance(vision._impl, _OpenAIVision)


def test_vision_unsupported_raises():
    provider = LanguageModelProvider(provider="cohere", model="m", api_key="k")
    with pytest.raises(ConfigurationError, match="Unsupported vision provider"):
        Vision(provider)
