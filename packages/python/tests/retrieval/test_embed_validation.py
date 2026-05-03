from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from rfnry_knowledge.knowledge.engine import KnowledgeEngine, _validate_query_text


def _stub_method_with_embeddings() -> SimpleNamespace:
    """Minimal stub of an ingestion method exposing ``_embeddings``."""
    return SimpleNamespace(_embeddings=MagicMock())


async def test_embed_single_rejects_oversize_text() -> None:
    engine = KnowledgeEngine.__new__(KnowledgeEngine)
    engine._initialized = True
    engine._config = SimpleNamespace(ingestion=SimpleNamespace(methods=[_stub_method_with_embeddings()]))  # type: ignore[assignment]
    with pytest.raises(ValueError, match="query exceeds"):
        await engine.embed_single("x" * 40_000)


async def test_embed_rejects_oversize_text() -> None:
    engine = KnowledgeEngine.__new__(KnowledgeEngine)
    engine._initialized = True
    engine._config = SimpleNamespace(ingestion=SimpleNamespace(methods=[_stub_method_with_embeddings()]))  # type: ignore[assignment]
    with pytest.raises(ValueError, match="query exceeds"):
        await engine.embed(["ok", "x" * 40_000])


def test_validate_query_text_raises_typed_input_error() -> None:
    from rfnry_knowledge.exceptions import InputError

    with pytest.raises(InputError, match="query exceeds"):
        _validate_query_text("x" * 40_000)
