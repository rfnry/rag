from rfnry_rag.providers import LanguageModelClient, LanguageModelProvider, build_registry
from rfnry_rag.providers.registry import _build_client_options


def test_build_client_options_includes_generation_params():
    provider = LanguageModelProvider(provider="openai", model="gpt-4o", api_key="sk-test")
    options = _build_client_options(provider, max_tokens=8192, temperature=0.7)
    assert options["model"] == "gpt-4o"
    assert options["max_tokens"] == 8192
    assert options["temperature"] == 0.7
    assert options["api_key"] == "sk-test"


def test_build_client_options_omits_api_key_when_none():
    provider = LanguageModelProvider(provider="openai", model="gpt-4o", api_key=None)
    options = _build_client_options(provider, max_tokens=4096, temperature=0.0)
    assert "api_key" not in options
    assert options["model"] == "gpt-4o"


def test_build_registry_primary_only():
    client = LanguageModelClient(
        provider=LanguageModelProvider(provider="openai", model="gpt-4o", api_key="sk-test"),
        max_tokens=8192,
        temperature=0.5,
    )
    registry = build_registry(client)
    from baml_py import ClientRegistry

    assert isinstance(registry, ClientRegistry)


def test_build_registry_with_fallback():
    client = LanguageModelClient(
        provider=LanguageModelProvider(provider="anthropic", model="claude-sonnet-4-20250514", api_key="ant-test"),
        fallback=LanguageModelProvider(provider="openai", model="gpt-4o", api_key="oai-test"),
        strategy="fallback",
        max_tokens=8192,
        temperature=0.3,
    )
    registry = build_registry(client)
    from baml_py import ClientRegistry

    assert isinstance(registry, ClientRegistry)


def test_build_registry_applies_same_generation_params_to_fallback(monkeypatch):
    """Fallback provider should receive the SAME max_tokens/temperature as primary."""
    captured_options = []

    from rfnry_rag.providers import registry as lm_module

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

    client = LanguageModelClient(
        provider=LanguageModelProvider(provider="anthropic", model="claude-sonnet-4-20250514", api_key="ant-test"),
        fallback=LanguageModelProvider(provider="openai", model="gpt-4o", api_key="oai-test"),
        strategy="fallback",
        max_tokens=9999,
        temperature=0.42,
    )
    build_registry(client)

    # Both primary and fallback should have been built with max_tokens=9999, temperature=0.42
    assert len(captured_options) == 2
    assert captured_options[0]["provider_model"] == "claude-sonnet-4-20250514"
    assert captured_options[0]["max_tokens"] == 9999
    assert captured_options[0]["temperature"] == 0.42
    assert captured_options[1]["provider_model"] == "gpt-4o"
    assert captured_options[1]["max_tokens"] == 9999
    assert captured_options[1]["temperature"] == 0.42


def test_language_model_client_default_timeout():
    client = LanguageModelClient(
        provider=LanguageModelProvider(provider="openai", model="m", api_key="k"),
    )
    assert client.timeout_seconds == 60


def test_language_model_client_rejects_non_positive_timeout():
    import pytest

    from rfnry_rag.exceptions import ConfigurationError

    with pytest.raises(ConfigurationError, match="timeout"):
        LanguageModelClient(
            provider=LanguageModelProvider(provider="openai", model="m", api_key="k"),
            timeout_seconds=0,
        )
    with pytest.raises(ConfigurationError, match="timeout"):
        LanguageModelClient(
            provider=LanguageModelProvider(provider="openai", model="m", api_key="k"),
            timeout_seconds=-5,
        )


def test_build_client_options_includes_timeout():
    provider = LanguageModelProvider(provider="openai", model="gpt-4o", api_key="k")
    options = _build_client_options(provider, max_tokens=4096, temperature=0.0, timeout_seconds=30)
    assert options.get("timeout") == 30 or options.get("request_timeout") == 30
