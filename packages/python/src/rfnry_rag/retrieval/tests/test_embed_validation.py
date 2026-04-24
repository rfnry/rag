from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from rfnry_rag.retrieval.server import RagEngine


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
