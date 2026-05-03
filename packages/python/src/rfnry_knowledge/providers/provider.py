from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import SecretStr

from rfnry_knowledge.exceptions import ConfigurationError

_MAX_RETRIES_LIMIT = 5


@dataclass(frozen=True)
class ProviderClient:
    name: str
    model: str
    api_key: SecretStr
    options: dict[str, Any] = field(default_factory=dict)
    max_retries: int = 3
    max_tokens: int = 4096  # unbounded: user pays for tokens; provider enforces its own context-window cap
    temperature: float = 0.0
    timeout_seconds: int = 60
    context_size: int | None = None
    fallback: ProviderClient | None = None
    strategy: Literal["primary_only", "fallback"] = "primary_only"
    boundary_api_key: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.name:
            raise ConfigurationError("ProviderClient.name must be a non-empty string")
        if not self.model:
            raise ConfigurationError("ProviderClient.model must be a non-empty string")
        if self.strategy not in ("primary_only", "fallback"):
            raise ConfigurationError(f"strategy must be 'primary_only' or 'fallback', got {self.strategy!r}")
        if self.strategy == "fallback" and self.fallback is None:
            raise ConfigurationError("strategy='fallback' requires a fallback ProviderClient")
        if not (0 <= self.max_retries <= _MAX_RETRIES_LIMIT):
            raise ConfigurationError(f"max_retries must be 0..{_MAX_RETRIES_LIMIT}, got {self.max_retries}")
        if self.timeout_seconds <= 0:
            raise ConfigurationError(f"timeout_seconds must be positive, got {self.timeout_seconds}")
        if not (0.0 <= self.temperature <= 2.0):
            raise ConfigurationError(
                f"temperature must be 0.0..2.0, got {self.temperature} — "
                "values > 2.0 produce incoherent output on all major providers"
            )
        if self.context_size is not None and self.context_size < 1:
            raise ConfigurationError(f"context_size must be >= 1 or None, got {self.context_size}")

    @property
    def display_name(self) -> str:
        return f"{self.name}:{self.model}"
