from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from baml_py import ClientRegistry

from rfnry_rag.common.errors import ConfigurationError

_MAX_RETRIES_LIMIT = 5

_CLIENT_DEFAULT = "Default"
_CLIENT_FALLBACK = "Fallback"
_CLIENT_ROUTER = "Router"


@dataclass
class LanguageModelProvider:
    """Identity: which API, which model, auth key.

    Used directly by Embeddings/Vision/Reranking for dedicated provider APIs,
    or wrapped by LanguageModelClient for BAML-routed LLM calls.
    """

    provider: str
    model: str
    api_key: str | None = None


@dataclass
class LanguageModelClient:
    """BAML-backed LLM client: routing, retries, fallback, generation params.

    max_tokens and temperature apply to both primary and fallback clients —
    per-fallback overrides are intentionally not supported.
    """

    provider: LanguageModelProvider
    fallback: LanguageModelProvider | None = None
    max_retries: int = 3
    strategy: Literal["primary_only", "fallback"] = "primary_only"
    max_tokens: int = 4096
    temperature: float = 0.0
    boundary_api_key: str | None = None

    def __post_init__(self) -> None:
        if self.strategy not in ("primary_only", "fallback"):
            raise ConfigurationError(f"Invalid strategy {self.strategy!r}, must be 'primary_only' or 'fallback'")
        if self.max_retries < 0 or self.max_retries > _MAX_RETRIES_LIMIT:
            raise ConfigurationError(f"max_retries must be 0-{_MAX_RETRIES_LIMIT}, got {self.max_retries}")
        if self.strategy == "fallback" and self.fallback is None:
            raise ConfigurationError("strategy='fallback' requires a fallback client")


def _retry_policy_name(max_retries: int) -> str | None:
    if max_retries == 0:
        return None
    return f"Retry{max_retries}"


def _build_client_options(provider: LanguageModelProvider, max_tokens: int, temperature: float) -> dict:
    options = {
        "model": provider.model,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if provider.api_key:
        options["api_key"] = provider.api_key
    return options


def build_registry(client: LanguageModelClient) -> ClientRegistry:
    registry = ClientRegistry()
    policy = _retry_policy_name(client.max_retries)

    registry.add_llm_client(
        _CLIENT_DEFAULT,
        provider=client.provider.provider,
        options=_build_client_options(client.provider, client.max_tokens, client.temperature),
        retry_policy=policy,
    )

    if client.strategy == "fallback" and client.fallback is not None:
        registry.add_llm_client(
            _CLIENT_FALLBACK,
            provider=client.fallback.provider,
            options=_build_client_options(client.fallback, client.max_tokens, client.temperature),
            retry_policy=policy,
        )
        registry.add_llm_client(
            _CLIENT_ROUTER,
            provider="fallback",
            options={"strategy": [_CLIENT_DEFAULT, _CLIENT_FALLBACK]},
        )
        registry.set_primary(_CLIENT_ROUTER)
    else:
        registry.set_primary(_CLIENT_DEFAULT)

    if client.boundary_api_key:
        os.environ["BOUNDARY_API_KEY"] = client.boundary_api_key

    return registry
