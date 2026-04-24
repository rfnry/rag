from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Literal

from baml_py import ClientRegistry

from rfnry_rag.common.errors import ConfigurationError

_boundary_logger = logging.getLogger("rfnry_rag.common.language_model")
_BOUNDARY_ENV = "BOUNDARY_API_KEY"

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
    api_key: str | None = field(default=None, repr=False)


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
    boundary_api_key: str | None = field(default=None, repr=False)
    # Per-call timeout. BAML's retry loop has no implicit timeout; without this,
    # a single hung LLM call (rate-limit stall, network partition) blocks the
    # event loop until the OS kills the socket.
    timeout_seconds: int = 60

    def __post_init__(self) -> None:
        if self.strategy not in ("primary_only", "fallback"):
            raise ConfigurationError(f"Invalid strategy {self.strategy!r}, must be 'primary_only' or 'fallback'")
        if self.max_retries < 0 or self.max_retries > _MAX_RETRIES_LIMIT:
            raise ConfigurationError(f"max_retries must be 0-{_MAX_RETRIES_LIMIT}, got {self.max_retries}")
        if self.strategy == "fallback" and self.fallback is None:
            raise ConfigurationError("strategy='fallback' requires a fallback client")
        if self.timeout_seconds <= 0:
            raise ConfigurationError(f"timeout_seconds must be positive, got {self.timeout_seconds}")


def _retry_policy_name(max_retries: int) -> str | None:
    if max_retries == 0:
        return None
    return f"Retry{max_retries}"


def _build_client_options(
    provider: LanguageModelProvider,
    max_tokens: int,
    temperature: float,
    timeout_seconds: int | None = None,
) -> dict:
    options: dict = {
        "model": provider.model,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if provider.api_key:
        options["api_key"] = provider.api_key
    if timeout_seconds is not None:
        # BAML passes these through to the underlying provider SDK.
        # Both keys are accepted by different provider backends — set both.
        options["timeout"] = timeout_seconds
        options["request_timeout"] = timeout_seconds
    return options


def build_registry(client: LanguageModelClient) -> ClientRegistry:
    registry = ClientRegistry()
    policy = _retry_policy_name(client.max_retries)

    registry.add_llm_client(
        _CLIENT_DEFAULT,
        provider=client.provider.provider,
        options=_build_client_options(client.provider, client.max_tokens, client.temperature, client.timeout_seconds),
        retry_policy=policy,
    )

    if client.strategy == "fallback" and client.fallback is not None:
        registry.add_llm_client(
            _CLIENT_FALLBACK,
            provider=client.fallback.provider,
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
        "language model client: provider=%s model=%s strategy=%s max_retries=%d timeout=%ds fallback=%s",
        client.provider.provider,
        client.provider.model,
        client.strategy,
        client.max_retries,
        client.timeout_seconds,
        bool(client.fallback),
    )

    return registry


def _apply_boundary_api_key(key: str | None) -> None:
    """Boundary authentication is a process-global env var (BOUNDARY_API_KEY)
    — BAML's collector reads it at send time. To avoid silent clobbering
    across multiple LanguageModelClient instances, we first-write-wins and
    raise on collision so a multi-tenant misconfiguration is never hidden."""
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
            "LanguageModelClient instances) to avoid this."
        )
