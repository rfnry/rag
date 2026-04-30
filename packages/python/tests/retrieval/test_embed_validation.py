from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from rfnry_rag.server import RagEngine, _validate_query_text


async def test_embed_single_rejects_oversize_text() -> None:
    rag = RagEngine.__new__(RagEngine)
    rag._initialized = True
    rag._config = SimpleNamespace(ingestion=SimpleNamespace(embeddings=MagicMock()))  # type: ignore[assignment]
    with pytest.raises(ValueError, match="query exceeds"):
        await rag.embed_single("x" * 40_000)


async def test_embed_rejects_oversize_text() -> None:
    rag = RagEngine.__new__(RagEngine)
    rag._initialized = True
    rag._config = SimpleNamespace(ingestion=SimpleNamespace(embeddings=MagicMock()))  # type: ignore[assignment]
    with pytest.raises(ValueError, match="query exceeds"):
        await rag.embed(["ok", "x" * 40_000])


def test_validate_query_text_raises_typed_input_error() -> None:
    from rfnry_rag.exceptions import InputError

    with pytest.raises(InputError, match="query exceeds"):
        _validate_query_text("x" * 40_000)
