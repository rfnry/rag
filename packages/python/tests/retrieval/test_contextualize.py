"""LLM-driven situating context per chunk — BAML-routed.

After the provider-decoupling refactor, ``contextualize_chunks_with_llm`` calls
the BAML ``SituateChunk`` function via ``instrument_baml_call``. These tests pin
the orchestrator behavior + ``ContextualChunkConfig`` invariants; per-vendor
dispatch is owned by BAML and is no longer exercised here.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from pydantic import SecretStr

from rfnry_knowledge.config import ContextualChunkConfig
from rfnry_knowledge.exceptions import ConfigurationError, EnrichmentSkipped, IngestionError
from rfnry_knowledge.ingestion.chunk.contextualize import contextualize_chunks_with_llm
from rfnry_knowledge.ingestion.models import ChunkedContent
from rfnry_knowledge.providers import ProviderClient


class _Counter:
    name = "test"
    model = "test"

    def count(self, text: str) -> int:
        return len(text.split())


def _make_chunk(idx: int, content: str = "", context: str = "") -> ChunkedContent:
    return ChunkedContent(
        content=content or f"passage_{idx}",
        chunk_index=idx,
        context=context,
        contextualized=f"{context}\n\n{content or f'passage_{idx}'}".strip(),
    )


def _client() -> ProviderClient:
    return ProviderClient(name="anthropic", model="claude-test", api_key=SecretStr("k"))


def _config(enabled: bool = True) -> ContextualChunkConfig:
    if not enabled:
        return ContextualChunkConfig()
    return ContextualChunkConfig(
        enabled=True,
        provider_client=_client(),
        token_counter=_Counter(),
    )


# ---------------------------------------------------------------------------
# ContextualChunkConfig — invariants
# ---------------------------------------------------------------------------


def test_config_disabled_by_default() -> None:
    cfg = ContextualChunkConfig()
    assert not cfg.enabled
    assert cfg.provider_client is None
    assert cfg.token_counter is None


def test_config_enabled_requires_provider_client() -> None:
    with pytest.raises(ConfigurationError, match="provider_client"):
        ContextualChunkConfig(enabled=True, token_counter=_Counter())


def test_config_enabled_requires_token_counter() -> None:
    with pytest.raises(ConfigurationError, match="token_counter"):
        ContextualChunkConfig(enabled=True, provider_client=_client())


def test_config_concurrency_bounds() -> None:
    with pytest.raises(ConfigurationError, match="concurrency"):
        ContextualChunkConfig(enabled=True, provider_client=_client(), token_counter=_Counter(), concurrency=0)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


async def test_disabled_returns_chunks_unchanged() -> None:
    cfg = ContextualChunkConfig()  # enabled=False
    chunks = [_make_chunk(0)]
    result = await contextualize_chunks_with_llm(chunks, document_text="doc", config=cfg)
    assert result is chunks


async def test_empty_chunks_short_circuits() -> None:
    cfg = _config(enabled=True)
    result = await contextualize_chunks_with_llm([], document_text="doc", config=cfg)
    assert result == []


async def test_oversized_document_skips() -> None:
    counter = _Counter()
    cfg = ContextualChunkConfig(
        enabled=True,
        provider_client=ProviderClient(
            name="anthropic", model="m", api_key=SecretStr("k"), context_size=1_000
        ),
        token_counter=counter,
        max_context_tokens=100,
    )
    long_doc = "word " * 200_000  # huge
    with pytest.raises(EnrichmentSkipped):
        await contextualize_chunks_with_llm([_make_chunk(0)], document_text=long_doc, config=cfg)


async def test_situates_chunk_via_baml() -> None:
    cfg = _config(enabled=True)
    chunk = _make_chunk(0, content="passage")
    with patch(
        "rfnry_knowledge.ingestion.chunk.contextualize.instrument_baml_call",
        new=AsyncMock(return_value="situating context blob"),
    ):
        result = await contextualize_chunks_with_llm([chunk], document_text="doc body", config=cfg)
    assert result[0].situating_context == "situating context blob"
    assert "situating context blob" in result[0].contextualized


async def test_failure_raises_ingestion_error() -> None:
    cfg = _config(enabled=True)
    chunk = _make_chunk(0)
    with patch(
        "rfnry_knowledge.ingestion.chunk.contextualize.instrument_baml_call",
        new=AsyncMock(side_effect=RuntimeError("boom")),
    ), pytest.raises(IngestionError):
        await contextualize_chunks_with_llm([chunk], document_text="doc", config=cfg)
