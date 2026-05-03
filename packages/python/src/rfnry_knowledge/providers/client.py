from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Literal

from rfnry_knowledge.exceptions import ConfigurationError
from rfnry_knowledge.providers.provider import ModelProvider

_MAX_RETRIES_LIMIT = 5


@dataclass
class LLMClient:
    provider: ModelProvider
    fallback: ModelProvider | None = None
    max_retries: int = 3
    strategy: Literal["primary_only", "fallback"] = "primary_only"
    max_tokens: int = 4096  # unbounded: user pays for tokens; provider enforces its own context-window cap
    temperature: float = 0.0
    boundary_api_key: str | None = field(default=None, repr=False)
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

    async def generate_text(self, system_prompt: str, history: str, user: str) -> str:
        from rfnry_knowledge.providers.text_generation import generate_text as _generate

        try:
            return await _generate(
                provider=self.provider,
                system_prompt=system_prompt,
                history=history,
                user=user,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                max_retries=self.max_retries,
                timeout_seconds=self.timeout_seconds,
            )
        except Exception:
            if self.strategy == "fallback" and self.fallback is not None:
                return await _generate(
                    provider=self.fallback,
                    system_prompt=system_prompt,
                    history=history,
                    user=user,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    max_retries=self.max_retries,
                    timeout_seconds=self.timeout_seconds,
                )
            raise

    def generate_text_stream(self, system_prompt: str, history: str, user: str) -> AsyncIterator[str]:
        from rfnry_knowledge.providers.text_generation import stream_text as _stream

        return _stream(
            provider=self.provider,
            system_prompt=system_prompt,
            history=history,
            user=user,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            max_retries=self.max_retries,
            timeout_seconds=self.timeout_seconds,
        )
