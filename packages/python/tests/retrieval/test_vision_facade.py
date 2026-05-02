import pytest

from rfnry_rag.exceptions import ConfigurationError
from rfnry_rag.ingestion.vision.anthropic import _AnthropicVision
from rfnry_rag.ingestion.vision.gemini import _GeminiVision
from rfnry_rag.ingestion.vision.openai import _OpenAIVision
from rfnry_rag.providers import (
    AnthropicModelProvider,
    CohereModelProvider,
    GoogleModelProvider,
    OpenAIModelProvider,
    Vision,
)


def test_vision_dispatches_to_anthropic():
    vision = Vision(AnthropicModelProvider(api_key="sk-test", model="claude-sonnet-4-20250514"))
    assert isinstance(vision._impl, _AnthropicVision)


def test_vision_dispatches_to_openai():
    vision = Vision(OpenAIModelProvider(api_key="sk-test", model="gpt-4o"))
    assert isinstance(vision._impl, _OpenAIVision)


def test_vision_facade_dispatches_to_google():
    vision = Vision(GoogleModelProvider(api_key="x", model="gemini-2.5-flash"))
    assert isinstance(vision._impl, _GeminiVision)


def test_vision_unsupported_raises():
    with pytest.raises(ConfigurationError, match="Unsupported vision provider"):
        Vision(CohereModelProvider(api_key="k", model="m"))


def test_vision_facade_unsupported_provider_lists_google():
    with pytest.raises(ConfigurationError, match="google"):
        Vision(CohereModelProvider(api_key="k", model="m"))
