from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Literal

from rfnry_rag.exceptions import ConfigurationError
from rfnry_rag.providers.provider import LanguageModelProvider

_MAX_RETRIES_LIMIT = 5


@dataclass
class LanguageModelClient:
    """LLM client: routing, retries, fallback, generation params.

    Used both as a BAML registry source (for structured-output BAML functions)
    and as the entry point for direct-SDK plain-text generation
    (``generate_text`` / ``generate_text_stream``).

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
    # Per-call timeout. Without this a single hung LLM call (rate-limit stall,
    # network partition) blocks the event loop until the OS kills the socket.
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
        """Plain-text generation against the primary provider, with optional fallback.

        Bypasses BAML — calls the provider SDK directly. Used for free-form
        outputs where BAML's structured-output parsing adds nothing.
        """
        from rfnry_rag.providers.text_generation import generate_text as _generate

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
        """Streaming counterpart of ``generate_text``; yields text deltas.

        Fallback is intentionally not applied to the streaming path: switching
        providers mid-stream would emit fragments from two different models
        and corrupt the output. Operators relying on fallback for streaming
        should retry the whole stream client-side on terminal error.
        """
        from rfnry_rag.providers.text_generation import stream_text as _stream

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
