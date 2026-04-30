from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from rfnry_rag.exceptions import ParseError
from rfnry_rag.ingestion.vision.constants import MAX_VISION_FILE_SIZE
from rfnry_rag.ingestion.vision.gemini import _GeminiVision
from rfnry_rag.providers import LanguageModel


def _make_vision() -> _GeminiVision:
    provider = LanguageModel(provider="gemini", model="gemini-2.5-flash", api_key="x")
    return _GeminiVision(provider)


def _png_bytes() -> bytes:
    # 1x1 transparent PNG header bytes — content-agnostic, just needs to be a real file.
    return (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4"
        b"\x89\x00\x00\x00\rIDATx\x9cc\xfc\xff\xff?\x00\x05\xfe\x02\xfe"
        b"\xa3\x35\x81\x84\x00\x00\x00\x00IEND\xaeB`\x82"
    )


async def test_gemini_vision_parse_minimal_image(tmp_path: Path) -> None:
    img = tmp_path / "sample.png"
    img.write_bytes(_png_bytes())

    vision = _make_vision()
    vision._client = SimpleNamespace(  # type: ignore[assignment]
        aio=SimpleNamespace(
            models=SimpleNamespace(generate_content=AsyncMock(return_value=SimpleNamespace(text="extracted content")))
        )
    )

    pages = await vision.parse(str(img))

    assert len(pages) == 1
    assert pages[0].page_number == 1
    assert pages[0].content == "extracted content"
    assert pages[0].metadata["source_type"] == "vision"
    assert pages[0].metadata["vision_model"] == "gemini-2.5-flash"
    assert pages[0].metadata["media_type"] == "image/png"
    assert pages[0].metadata["char_count"] == len("extracted content")


async def test_gemini_vision_rejects_oversize_file(tmp_path: Path) -> None:
    img = tmp_path / "big.png"
    img.write_bytes(b"\x00" * (MAX_VISION_FILE_SIZE + 1))

    vision = _make_vision()
    vision._client = SimpleNamespace(  # type: ignore[assignment]
        aio=SimpleNamespace(models=SimpleNamespace(generate_content=AsyncMock()))
    )

    with pytest.raises(ParseError, match="too large"):
        await vision.parse(str(img))


async def test_gemini_vision_rejects_unsupported_extension(tmp_path: Path) -> None:
    bad = tmp_path / "notes.txt"
    bad.write_text("plain text")

    vision = _make_vision()

    with pytest.raises(ValueError, match="Unsupported image type"):
        await vision.parse(str(bad))


async def test_gemini_vision_handles_empty_response(tmp_path: Path) -> None:
    img = tmp_path / "sample.png"
    img.write_bytes(_png_bytes())

    vision = _make_vision()
    vision._client = SimpleNamespace(  # type: ignore[assignment]
        aio=SimpleNamespace(models=SimpleNamespace(generate_content=AsyncMock(return_value=SimpleNamespace(text=""))))
    )

    with pytest.raises(ValueError, match="empty content"):
        await vision.parse(str(img))


async def test_gemini_vision_metadata_includes_provider_tag(tmp_path: Path) -> None:
    img = tmp_path / "sample.png"
    img.write_bytes(_png_bytes())

    vision = _make_vision()
    vision._client = SimpleNamespace(  # type: ignore[assignment]
        aio=SimpleNamespace(models=SimpleNamespace(generate_content=AsyncMock(return_value=SimpleNamespace(text="ok"))))
    )

    pages = await vision.parse(str(img))

    assert pages[0].metadata["vision_provider"] == "gemini"
