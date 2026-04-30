import pytest

from rfnry_rag.exceptions import ConfigurationError
from rfnry_rag.ingestion.vision.anthropic import _AnthropicVision
from rfnry_rag.ingestion.vision.gemini import _GeminiVision
from rfnry_rag.ingestion.vision.openai import _OpenAIVision
from rfnry_rag.providers import LanguageModel, Vision


def test_vision_dispatches_to_anthropic():
    provider = LanguageModel(provider="anthropic", model="claude-sonnet-4-20250514", api_key="sk-test")
    vision = Vision(provider)
    assert isinstance(vision._impl, _AnthropicVision)


def test_vision_dispatches_to_openai():
    provider = LanguageModel(provider="openai", model="gpt-4o", api_key="sk-test")
    vision = Vision(provider)
    assert isinstance(vision._impl, _OpenAIVision)


def test_vision_facade_dispatches_to_gemini():
    provider = LanguageModel(provider="gemini", model="gemini-2.5-flash", api_key="x")
    vision = Vision(provider)
    assert isinstance(vision._impl, _GeminiVision)


def test_vision_unsupported_raises():
    provider = LanguageModel(provider="cohere", model="m", api_key="k")
    with pytest.raises(ConfigurationError, match="Unsupported vision provider"):
        Vision(provider)


def test_vision_facade_unsupported_provider_lists_gemini():
    provider = LanguageModel(provider="cohere", model="m", api_key="k")
    with pytest.raises(ConfigurationError, match="gemini"):
        Vision(provider)
