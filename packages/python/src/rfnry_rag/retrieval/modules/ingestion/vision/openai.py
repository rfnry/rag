import base64
from pathlib import Path

from openai import AsyncOpenAI

from rfnry_rag.common.language_model import LanguageModelProvider
from rfnry_rag.retrieval.common.errors import ParseError
from rfnry_rag.retrieval.common.logging import get_logger
from rfnry_rag.retrieval.modules.ingestion.models import ParsedPage
from rfnry_rag.retrieval.modules.ingestion.vision.constants import (
    MAX_VISION_FILE_SIZE,
    MEDIA_TYPES,
    VISION_EXTRACTION_PROMPT,
)

logger = get_logger(__name__)


class _OpenAIVision:
    def __init__(
        self,
        provider: LanguageModelProvider,
        max_tokens: int = 4096,
        max_retries: int = 3,
    ) -> None:
        self._client = AsyncOpenAI(api_key=provider.api_key, max_retries=max_retries)
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

        image_data = base64.standard_b64encode(path.read_bytes()).decode("utf-8")

        response = await self._client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{media_type};base64,{image_data}"},
                        },
                        {"type": "text", "text": VISION_EXTRACTION_PROMPT},
                    ],
                }
            ],
        )

        content = response.choices[0].message.content

        if not content:
            raise ValueError(f"Vision model returned empty content for '{path.name}' (possible content policy refusal)")

        return [
            ParsedPage(
                page_number=1,
                content=content,
                metadata={
                    "source_type": "vision",
                    "vision_model": self._model,
                    "vision_provider": "openai",
                    "media_type": media_type,
                    "char_count": len(content),
                },
            )
        ]
