from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from rfnry_rag.exceptions import ConfigurationError
from rfnry_rag.providers.provider import LanguageModelProvider

_MAX_RETRIES_LIMIT = 5


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
    max_tokens: int = 4096  # unbounded: user pays for tokens; provider enforces its own context-window cap
    temperature: float = 0.0
    boundary_api_key: str | None = field(default=None, repr=False)
    # Per-call timeout. BAML's retry loop has no implicit timeout; without this,
    # a single hung LLM call (rate-limit stall, network partition) blocks the
    # event loop until the OS kills the socket.
    timeout_seconds: int = 60  # unbounded: operator sets based on provider SLA; no universal ceiling applies

    def __post_init__(self) -> None:
        if self.strategy not in ("primary_only", "fallback"):
            raise ConfigurationError(f"Invalid strategy {self.strategy!r}, must be 'primary_only' or 'fallback'")
        if self.max_retries < 0 or self.max_retries > _MAX_RETRIES_LIMIT:
            raise ConfigurationError(f"max_retries must be 0-{_MAX_RETRIES_LIMIT}, got {self.max_retries}")
        if self.strategy == "fallback" and self.fallback is None:
            raise ConfigurationError("strategy='fallback' requires a fallback client")
        if self.timeout_seconds <= 0:
            raise ConfigurationError(f"timeout_seconds must be positive, got {self.timeout_seconds}")
        if not (0.0 <= self.temperature <= 2.0):
            raise ConfigurationError(
                f"temperature must be between 0.0 and 2.0, got {self.temperature} — "
                "values > 2.0 produce incoherent output on all major providers"
            )
