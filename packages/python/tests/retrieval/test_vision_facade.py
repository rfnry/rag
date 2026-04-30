import pytest

from rfnry_rag.common.language_model import LanguageModelProvider
from rfnry_rag.exceptions import ConfigurationError
from rfnry_rag.ingestion.vision.anthropic import _AnthropicVision
from rfnry_rag.ingestion.vision.facade import Vision
from rfnry_rag.ingestion.vision.gemini import _GeminiVision
from rfnry_rag.ingestion.vision.openai import _OpenAIVision


def test_vision_dispatches_to_anthropic():
    provider = LanguageModelProvider(provider="anthropic", model="claude-sonnet-4-20250514", api_key="sk-test")
    vision = Vision(provider)
    assert isinstance(vision._impl, _AnthropicVision)


def test_vision_dispatches_to_openai():
    provider = LanguageModelProvider(provider="openai", model="gpt-4o", api_key="sk-test")
    vision = Vision(provider)
    assert isinstance(vision._impl, _OpenAIVision)


def test_vision_facade_dispatches_to_gemini():
    provider = LanguageModelProvider(provider="gemini", model="gemini-2.5-flash", api_key="x")
    vision = Vision(provider)
    assert isinstance(vision._impl, _GeminiVision)


def test_vision_unsupported_raises():
    provider = LanguageModelProvider(provider="cohere", model="m", api_key="k")
    with pytest.raises(ConfigurationError, match="Unsupported vision provider"):
        Vision(provider)


def test_vision_facade_unsupported_provider_lists_gemini():
    provider = LanguageModelProvider(provider="cohere", model="m", api_key="k")
    with pytest.raises(ConfigurationError, match="gemini"):
        Vision(provider)
