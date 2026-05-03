from rfnry_knowledge.providers import (
    AnthropicModelProvider,
    LLMClient,
    OpenAIModelProvider,
    build_registry,
)
from rfnry_knowledge.providers.registry import _build_client_options


def _openai(api_key: str = "sk-test", model: str = "gpt-4o") -> OpenAIModelProvider:
    return OpenAIModelProvider(api_key=api_key, model=model)


def _anthropic(api_key: str = "ant-test", model: str = "claude-sonnet-4-20250514") -> AnthropicModelProvider:
    return AnthropicModelProvider(api_key=api_key, model=model)


def test_build_client_options_includes_generation_params():
    options = _build_client_options(_openai(), max_tokens=8192, temperature=0.7)
    assert options["model"] == "gpt-4o"
    assert options["max_tokens"] == 8192
    assert options["temperature"] == 0.7
    assert options["api_key"] == "sk-test"


def test_build_registry_primary_only():
    client = LLMClient(provider=_openai(), max_tokens=8192, temperature=0.5)
    registry = build_registry(client)
    from baml_py import ClientRegistry

    assert isinstance(registry, ClientRegistry)


def test_build_registry_with_fallback():
    client = LLMClient(
        provider=_anthropic(),
        fallback=_openai(api_key="oai-test"),
        strategy="fallback",
        max_tokens=8192,
        temperature=0.3,
    )
    registry = build_registry(client)
    from baml_py import ClientRegistry

    assert isinstance(registry, ClientRegistry)


def test_build_registry_applies_same_generation_params_to_fallback(monkeypatch):
    captured_options = []

    from rfnry_knowledge.providers import registry as lm_module

    original = lm_module._build_client_options

    def spy(provider, max_tokens, temperature, timeout_seconds=None):
        captured_options.append(
            {
                "provider_model": provider.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )
        return original(provider, max_tokens, temperature, timeout_seconds)

    monkeypatch.setattr(lm_module, "_build_client_options", spy)

    client = LLMClient(
        provider=_anthropic(),
        fallback=_openai(api_key="oai-test"),
        strategy="fallback",
        max_tokens=9999,
        temperature=0.42,
    )
    build_registry(client)

    assert len(captured_options) == 2
    assert captured_options[0]["provider_model"] == "claude-sonnet-4-20250514"
    assert captured_options[0]["max_tokens"] == 9999
    assert captured_options[0]["temperature"] == 0.42
    assert captured_options[1]["provider_model"] == "gpt-4o"
    assert captured_options[1]["max_tokens"] == 9999
    assert captured_options[1]["temperature"] == 0.42


def test_generative_model_client_default_timeout():
    client = LLMClient(provider=_openai(api_key="k", model="m"))
    assert client.timeout_seconds == 60


def test_generative_model_client_rejects_non_positive_timeout():
    import pytest

    from rfnry_knowledge.exceptions import ConfigurationError

    with pytest.raises(ConfigurationError, match="timeout"):
        LLMClient(provider=_openai(api_key="k", model="m"), timeout_seconds=0)
    with pytest.raises(ConfigurationError, match="timeout"):
        LLMClient(provider=_openai(api_key="k", model="m"), timeout_seconds=-5)


def test_build_client_options_includes_timeout():
    options = _build_client_options(_openai(api_key="k"), max_tokens=4096, temperature=0.0, timeout_seconds=30)
    assert options.get("timeout") == 30 or options.get("request_timeout") == 30
