from __future__ import annotations

import logging
import os

from baml_py import ClientRegistry

from rfnry_rag.exceptions import ConfigurationError
from rfnry_rag.providers.client import GenerativeModelClient
from rfnry_rag.providers.provider import ModelProvider

_boundary_logger = logging.getLogger("rfnry_rag.providers.registry")
_BOUNDARY_ENV = "BOUNDARY_API_KEY"

_CLIENT_DEFAULT = "Default"
_CLIENT_FALLBACK = "Fallback"
_CLIENT_ROUTER = "Router"


def _retry_policy_name(max_retries: int) -> str | None:
    if max_retries == 0:
        return None
    return f"Retry{max_retries}"


def _build_client_options(
    provider: ModelProvider,
    max_tokens: int,
    temperature: float,
    timeout_seconds: int | None = None,
) -> dict:
    options: dict = {
        "model": provider.model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "api_key": provider.api_key.get_secret_value(),
    }
    if timeout_seconds is not None:
        options["timeout"] = timeout_seconds
        options["request_timeout"] = timeout_seconds
    return options


def build_registry(client: GenerativeModelClient) -> ClientRegistry:
    registry = ClientRegistry()
    policy = _retry_policy_name(client.max_retries)

    registry.add_llm_client(
        _CLIENT_DEFAULT,
        provider=client.provider.kind,
        options=_build_client_options(client.provider, client.max_tokens, client.temperature, client.timeout_seconds),
        retry_policy=policy,
    )

    if client.strategy == "fallback" and client.fallback is not None:
        registry.add_llm_client(
            _CLIENT_FALLBACK,
            provider=client.fallback.kind,
            options=_build_client_options(
                client.fallback, client.max_tokens, client.temperature, client.timeout_seconds
            ),
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

    _apply_boundary_api_key(client.boundary_api_key)

    _boundary_logger.info(
        "generative model client: provider=%s model=%s strategy=%s max_retries=%d timeout=%ds fallback=%s",
        client.provider.kind,
        client.provider.model,
        client.strategy,
        client.max_retries,
        client.timeout_seconds,
        bool(client.fallback),
    )

    return registry


def _apply_boundary_api_key(key: str | None) -> None:
    if not key:
        return
    existing = os.environ.get(_BOUNDARY_ENV)
    if existing is None:
        os.environ[_BOUNDARY_ENV] = key
        return
    if existing != key:
        raise ConfigurationError(
            "boundary_api_key collision — a different BOUNDARY_API_KEY is "
            "already set for this process. Set BOUNDARY_API_KEY in the "
            "environment once at process start (shared across all "
            "GenerativeModelClient instances) to avoid this."
        )
