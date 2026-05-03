from pydantic import SecretStr

from rfnry_knowledge.providers import ProviderClient, build_registry
from rfnry_knowledge.providers.registry import _client_options


def _client(name: str = "openai", model: str = "gpt-4o", api_key: str = "sk-test", **kwargs) -> ProviderClient:
    return ProviderClient(name=name, model=model, api_key=SecretStr(api_key), **kwargs)


def test_client_options_includes_generation_params():
    client = _client(model="gpt-4o", api_key="sk-test", max_tokens=8192, temperature=0.7)
    options = _client_options(client)
    assert options["model"] == "gpt-4o"
    assert options["max_tokens"] == 8192
    assert options["temperature"] == 0.7
    assert options["api_key"] == "sk-test"


def test_build_registry_primary_only():
    registry = build_registry(_client(max_tokens=8192, temperature=0.5))
    from baml_py import ClientRegistry

    assert isinstance(registry, ClientRegistry)


def test_build_registry_with_fallback():
    primary = _client(name="anthropic", model="claude-sonnet-4-20250514")
    fallback = _client(name="openai", api_key="oai-test")
    registry = build_registry(
        ProviderClient(
            name="anthropic",
            model="claude-sonnet-4-20250514",
            api_key=SecretStr("ant-test"),
            fallback=fallback,
            strategy="fallback",
            max_tokens=8192,
            temperature=0.3,
        )
    )
    assert primary  # ensure helper is exercised
    from baml_py import ClientRegistry

    assert isinstance(registry, ClientRegistry)


def test_build_registry_applies_same_generation_params_to_fallback(monkeypatch):
    captured_options = []

    from rfnry_knowledge.providers import registry as lm_module

    original = lm_module._client_options

    def spy(client):
        captured_options.append(
            {
                "name": client.name,
                "model": client.model,
                "max_tokens": client.max_tokens,
                "temperature": client.temperature,
            }
        )
        return original(client)

    monkeypatch.setattr(lm_module, "_client_options", spy)

    fallback = _client(name="openai", api_key="oai-test", max_tokens=9999, temperature=0.42)
    build_registry(
        ProviderClient(
            name="anthropic",
            model="claude-sonnet-4-20250514",
            api_key=SecretStr("ant-test"),
            fallback=fallback,
            strategy="fallback",
            max_tokens=9999,
            temperature=0.42,
        )
    )

    assert len(captured_options) == 2
    assert captured_options[0]["name"] == "anthropic"
    assert captured_options[0]["max_tokens"] == 9999
    assert captured_options[0]["temperature"] == 0.42
    assert captured_options[1]["name"] == "openai"
    assert captured_options[1]["max_tokens"] == 9999
    assert captured_options[1]["temperature"] == 0.42


def test_provider_client_default_timeout():
    assert _client().timeout_seconds == 60


def test_provider_client_rejects_non_positive_timeout():
    import pytest

    from rfnry_knowledge.exceptions import ConfigurationError

    with pytest.raises(ConfigurationError, match="timeout"):
        _client(timeout_seconds=0)
    with pytest.raises(ConfigurationError, match="timeout"):
        _client(timeout_seconds=-5)


def test_client_options_includes_timeout():
    options = _client_options(_client(timeout_seconds=30))
    assert options.get("timeout") == 30 or options.get("request_timeout") == 30
