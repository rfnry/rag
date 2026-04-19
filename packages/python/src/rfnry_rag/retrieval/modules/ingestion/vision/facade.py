from __future__ import annotations

from rfnry_rag.common.errors import ConfigurationError
from rfnry_rag.common.language_model import LanguageModelProvider
from rfnry_rag.retrieval.modules.ingestion.models import ParsedPage
from rfnry_rag.retrieval.modules.ingestion.vision.anthropic import _AnthropicVision
from rfnry_rag.retrieval.modules.ingestion.vision.openai import _OpenAIVision


class Vision:
    """Vision client dispatching to the correct provider implementation."""

    def __init__(
        self,
        provider: LanguageModelProvider,
        max_tokens: int = 4096,
        max_retries: int = 3,
    ) -> None:
        match provider.provider:
            case "anthropic":
                self._impl: _AnthropicVision | _OpenAIVision = _AnthropicVision(
                    provider, max_tokens=max_tokens, max_retries=max_retries
                )
            case "openai":
                self._impl = _OpenAIVision(provider, max_tokens=max_tokens, max_retries=max_retries)
            case _:
                raise ConfigurationError(
                    f"Unsupported vision provider: {provider.provider!r}. Supported: anthropic, openai."
                )

    async def parse(self, file_path: str, pages: set[int] | None = None) -> list[ParsedPage]:
        return await self._impl.parse(file_path, pages)
