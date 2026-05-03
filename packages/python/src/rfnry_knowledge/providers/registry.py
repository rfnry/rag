from __future__ import annotations

import logging
import os
from typing import Any

from baml_py import ClientRegistry

from rfnry_knowledge.exceptions import ConfigurationError
from rfnry_knowledge.providers.provider import ProviderClient

_logger = logging.getLogger("rfnry_knowledge.providers.registry")
_BOUNDARY_ENV = "BOUNDARY_API_KEY"

_CLIENT_DEFAULT = "Default"
_CLIENT_FALLBACK = "Fallback"
_CLIENT_ROUTER = "Router"


def _retry_policy_name(max_retries: int) -> str | None:
    return None if max_retries == 0 else f"Retry{max_retries}"


def _client_options(client: ProviderClient) -> dict[str, Any]:
    options: dict[str, Any] = {
        "model": client.model,
        "api_key": client.api_key.get_secret_value(),
        "temperature": client.temperature,
        "max_tokens": client.max_tokens,
        "timeout": client.timeout_seconds,
        "request_timeout": client.timeout_seconds,
    }
    options.update(client.options)
    return options


def build_registry(client: ProviderClient) -> ClientRegistry:
    registry = ClientRegistry()
    policy = _retry_policy_name(client.max_retries)

    registry.add_llm_client(
        _CLIENT_DEFAULT,
        provider=client.name,
        options=_client_options(client),
        retry_policy=policy,
    )

    if client.strategy == "fallback" and client.fallback is not None:
        registry.add_llm_client(
            _CLIENT_FALLBACK,
            provider=client.fallback.name,
            options=_client_options(client.fallback),
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

    _logger.info(
        "provider client: name=%s model=%s strategy=%s max_retries=%d timeout=%ds fallback=%s",
        client.name,
        client.model,
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
            "ProviderClient instances) to avoid this."
        )
