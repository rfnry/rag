from pathlib import Path

from google import genai
from google.genai import types

from rfnry_rag.exceptions import ParseError
from rfnry_rag.ingestion.models import ParsedPage
from rfnry_rag.ingestion.vision.constants import (
    MAX_VISION_FILE_SIZE,
    MEDIA_TYPES,
    VISION_EXTRACTION_PROMPT,
)
from rfnry_rag.logging import get_logger
from rfnry_rag.providers.provider import LanguageModel

logger = get_logger(__name__)


class _GeminiVision:
    def __init__(
        self,
        provider: LanguageModel,
        max_tokens: int = 4096,
        max_retries: int = 3,
    ) -> None:
        # The google-genai SDK exposes retries via HttpOptions.retry_options.attempts
        # rather than as a direct Client kwarg, so we wire it through there.
        http_options = types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=max_retries))
        self._client = genai.Client(api_key=provider.api_key, http_options=http_options)
        self._model = provider.model
        self._max_tokens = max_tokens

    async def parse(self, file_path: str, pages: set[int] | None = None) -> list[ParsedPage]:
        path = Path(file_path)
        ext = path.suffix.lower()
        media_type = MEDIA_TYPES.get(ext)
        if not media_type:
            raise ValueError(f"Unsupported image type: {ext}. Supported: {list(MEDIA_TYPES.keys())}")

        if pages is not None and 1 not in pages:
            return []

        file_size = path.stat().st_size
        if file_size > MAX_VISION_FILE_SIZE:
            raise ParseError(
                f"Image file too large ({file_size / 1024 / 1024:.1f} MB). "
                f"Maximum allowed: {MAX_VISION_FILE_SIZE / 1024 / 1024:.0f} MB"
            )

        image_bytes = path.read_bytes()

        image_part = types.Part.from_bytes(data=image_bytes, mime_type=media_type)
        # mypy widens a heterogeneous Part+str literal list to list[object]; the SDK signature
        # accepts list[str | Image | File | Part], so the runtime types are correct.
        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=[image_part, VISION_EXTRACTION_PROMPT],  # type: ignore[arg-type]
            config=types.GenerateContentConfig(max_output_tokens=self._max_tokens),
        )

        content = response.text

        if not content:
            raise ValueError(f"Vision model returned empty content for '{path.name}' (possible content policy refusal)")

        return [
            ParsedPage(
                page_number=1,
                content=content,
                metadata={
                    "source_type": "vision",
                    "vision_model": self._model,
                    "vision_provider": "gemini",
                    "media_type": media_type,
                    "char_count": len(content),
                },
            )
        ]
