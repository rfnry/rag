"""LLM-driven situating context per chunk.

Pins the public contract of ``ContextualChunkConfig`` and the
``contextualize_chunks_with_llm`` orchestrator. Per-provider dispatch is
checked by patching the SDK at the import site inside
``ingestion.chunk.contextualize``.

Identifier hygiene: tests use abstract names so the fixtures themselves do
not seed bias-listed terms.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rfnry_rag.config import ContextualChunkConfig
from rfnry_rag.exceptions import ConfigurationError, IngestionError
from rfnry_rag.ingestion.chunk.contextualize import contextualize_chunks_with_llm
from rfnry_rag.ingestion.models import ChunkedContent
from rfnry_rag.providers import LanguageModel, LanguageModelClient


def _make_chunk(idx: int, content: str = "", context: str = "") -> ChunkedContent:
    return ChunkedContent(
        content=content or f"passage_{idx}",
        chunk_index=idx,
        context=context,
        contextualized=f"{context}\n\n{content or f'passage_{idx}'}".strip(),
    )


def _make_client(provider: str = "anthropic") -> LanguageModelClient:
    return LanguageModelClient(lm=LanguageModel(provider=provider, model="m1", api_key="k"))


# ---------------------------------------------------------------------------
# ContextualChunkConfig — bounds + invariants
# ---------------------------------------------------------------------------


def test_contextual_chunk_config_requires_lm_client_when_enabled() -> None:
    with pytest.raises(ConfigurationError, match="lm_client"):
        ContextualChunkConfig(enabled=True, lm_client=None)


def test_contextual_chunk_config_bounds_concurrency() -> None:
    with pytest.raises(ConfigurationError, match="concurrency"):
        ContextualChunkConfig(concurrency=0)
    with pytest.raises(ConfigurationError, match="concurrency"):
        ContextualChunkConfig(concurrency=101)
    for n in (1, 5, 100):
        cfg = ContextualChunkConfig(concurrency=n)
        assert cfg.concurrency == n


def test_contextual_chunk_config_bounds_max_context_tokens() -> None:
    with pytest.raises(ConfigurationError, match="max_context_tokens"):
        ContextualChunkConfig(max_context_tokens=9)
    with pytest.raises(ConfigurationError, match="max_context_tokens"):
        ContextualChunkConfig(max_context_tokens=501)
    for n in (10, 100, 500):
        cfg = ContextualChunkConfig(max_context_tokens=n)
        assert cfg.max_context_tokens == n


# ---------------------------------------------------------------------------
# Orchestrator — disabled / empty / unsupported provider
# ---------------------------------------------------------------------------


async def test_disabled_is_noop() -> None:
    chunks = [_make_chunk(0, content="alpha"), _make_chunk(1, content="beta")]
    cfg = ContextualChunkConfig(enabled=False)
    result = await contextualize_chunks_with_llm(chunks, document_text="doc", config=cfg)
    assert result is chunks
    assert all(c.situating_context == "" for c in chunks)


async def test_empty_chunks_is_noop() -> None:
    cfg = ContextualChunkConfig(enabled=True, lm_client=_make_client())
    result = await contextualize_chunks_with_llm([], document_text="doc", config=cfg)
    assert result == []


async def test_unsupported_provider_raises() -> None:
    chunks = [_make_chunk(0)]
    cfg = ContextualChunkConfig(
        enabled=True,
        lm_client=LanguageModelClient(lm=LanguageModel(provider="bedrock", model="m", api_key="k")),
    )
    with pytest.raises(IngestionError, match="chunk_index=0") as exc_info:
        await contextualize_chunks_with_llm(chunks, document_text="doc", config=cfg)
    cause = exc_info.value.__cause__
    assert isinstance(cause, ConfigurationError)
    assert "Supported: anthropic, openai, gemini" in str(cause)


# ---------------------------------------------------------------------------
# Anthropic dispatch — system block carries cache_control, chunk in user message
# ---------------------------------------------------------------------------


async def test_anthropic_dispatch_structure() -> None:
    captured: dict[str, object] = {}

    async def fake_create(**kwargs: object) -> object:
        captured.update(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(text="situated")])

    with patch("anthropic.AsyncAnthropic") as mock_cls, patch(
        "anthropic.types.TextBlock", new=SimpleNamespace
    ):
        fake_client = MagicMock()
        fake_client.messages.create = AsyncMock(side_effect=fake_create)
        mock_cls.return_value = fake_client

        chunks = [_make_chunk(0, content="alpha")]
        cfg = ContextualChunkConfig(
            enabled=True,
            lm_client=_make_client("anthropic"),
            max_context_tokens=80,
        )
        await contextualize_chunks_with_llm(chunks, document_text="DOCBODY", config=cfg)

    system = captured["system"]
    assert isinstance(system, list) and len(system) == 1
    block = system[0]
    assert block["type"] == "text"
    assert "DOCBODY" in block["text"]
    assert block["cache_control"] == {"type": "ephemeral"}
    messages = captured["messages"]
    assert isinstance(messages, list) and messages[0]["role"] == "user"
    assert "alpha" in messages[0]["content"]
    assert captured["max_tokens"] == 80
    assert chunks[0].situating_context == "situated"


# ---------------------------------------------------------------------------
# OpenAI dispatch — document body in system message
# ---------------------------------------------------------------------------


async def test_openai_dispatch_structure() -> None:
    captured: dict[str, object] = {}

    async def fake_create(**kwargs: object) -> object:
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="situated_openai"))]
        )

    with patch("openai.AsyncOpenAI") as mock_cls:
        fake_client = MagicMock()
        fake_client.chat.completions.create = AsyncMock(side_effect=fake_create)
        mock_cls.return_value = fake_client

        chunks = [_make_chunk(0, content="alpha"), _make_chunk(1, content="beta")]
        cfg = ContextualChunkConfig(
            enabled=True,
            lm_client=_make_client("openai"),
            max_context_tokens=60,
            concurrency=1,
        )
        await contextualize_chunks_with_llm(chunks, document_text="DOCBODY", config=cfg)

    messages = captured["messages"]
    assert isinstance(messages, list) and len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "DOCBODY" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert captured["max_tokens"] == 60
    assert chunks[0].situating_context == "situated_openai"


# ---------------------------------------------------------------------------
# Gemini dispatch — document body in system_instruction
# ---------------------------------------------------------------------------


async def test_gemini_dispatch_structure() -> None:
    captured: dict[str, object] = {}

    async def fake_generate(*, model: str, contents: str, config: object) -> object:
        captured["model"] = model
        captured["contents"] = contents
        captured["config"] = config
        return SimpleNamespace(text="situated_gemini")

    with patch("google.genai.Client") as mock_cls:
        fake_client = MagicMock()
        fake_client.aio.models.generate_content = AsyncMock(side_effect=fake_generate)
        mock_cls.return_value = fake_client

        chunks = [_make_chunk(0, content="alpha")]
        cfg = ContextualChunkConfig(
            enabled=True,
            lm_client=_make_client("gemini"),
            max_context_tokens=70,
        )
        await contextualize_chunks_with_llm(chunks, document_text="DOCBODY", config=cfg)

    cfg_obj = captured["config"]
    assert "DOCBODY" in cfg_obj.system_instruction
    assert "alpha" in captured["contents"]
    assert cfg_obj.max_output_tokens == 70
    assert chunks[0].situating_context == "situated_gemini"


# ---------------------------------------------------------------------------
# Per-chunk failure wrapping
# ---------------------------------------------------------------------------


async def test_per_chunk_failure_raises_with_chunk_index() -> None:
    with patch("anthropic.AsyncAnthropic") as mock_cls:
        fake_client = MagicMock()
        fake_client.messages.create = AsyncMock(side_effect=RuntimeError("boom"))
        mock_cls.return_value = fake_client

        chunks = [_make_chunk(7, content="alpha")]
        cfg = ContextualChunkConfig(enabled=True, lm_client=_make_client("anthropic"))
        with pytest.raises(IngestionError, match="chunk_index=7") as exc_info:
            await contextualize_chunks_with_llm(chunks, document_text="d", config=cfg)
        assert isinstance(exc_info.value.__cause__, RuntimeError)


# ---------------------------------------------------------------------------
# Concurrency cap
# ---------------------------------------------------------------------------


async def test_concurrency_cap_respected() -> None:
    in_flight = 0
    max_in_flight = 0
    lock = asyncio.Lock()
    call_count = 0

    async def fake_create(**kwargs: object) -> object:
        nonlocal in_flight, max_in_flight, call_count
        async with lock:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            call_count += 1
        await asyncio.sleep(0.01)
        async with lock:
            in_flight -= 1
        return SimpleNamespace(content=[SimpleNamespace(text="x")])

    with patch("anthropic.AsyncAnthropic") as mock_cls, patch(
        "anthropic.types.TextBlock", new=SimpleNamespace
    ):
        fake_client = MagicMock()
        fake_client.messages.create = AsyncMock(side_effect=fake_create)
        mock_cls.return_value = fake_client

        chunks = [_make_chunk(i, content=f"p{i}") for i in range(10)]
        cfg = ContextualChunkConfig(
            enabled=True,
            lm_client=_make_client("anthropic"),
            concurrency=3,
        )
        await contextualize_chunks_with_llm(chunks, document_text="d", config=cfg)

    assert call_count == 10
    assert max_in_flight <= 3
    assert max_in_flight >= 2  # sanity: actually parallelised


# ---------------------------------------------------------------------------
# Composition — situating context folded into contextualized
# ---------------------------------------------------------------------------


async def test_situating_context_is_folded_into_contextualized() -> None:
    async def fake_create(**kwargs: object) -> object:
        return SimpleNamespace(content=[SimpleNamespace(text="SITUATED_BLOB")])

    with patch("anthropic.AsyncAnthropic") as mock_cls, patch(
        "anthropic.types.TextBlock", new=SimpleNamespace
    ):
        fake_client = MagicMock()
        fake_client.messages.create = AsyncMock(side_effect=fake_create)
        mock_cls.return_value = fake_client

        chunks = [
            _make_chunk(0, content="alpha-content", context="Document: Foo | Page: 1"),
        ]
        cfg = ContextualChunkConfig(enabled=True, lm_client=_make_client("anthropic"))
        await contextualize_chunks_with_llm(chunks, document_text="d", config=cfg)

    folded = chunks[0].contextualized
    assert "Document: Foo | Page: 1" in folded
    assert "SITUATED_BLOB" in folded
    assert "alpha-content" in folded
    assert folded.index("Document: Foo") < folded.index("SITUATED_BLOB") < folded.index("alpha-content")


async def test_fold_skips_empty_structural_header() -> None:
    async def fake_create(**kwargs: object) -> object:
        return SimpleNamespace(content=[SimpleNamespace(text="BLOB")])

    with patch("anthropic.AsyncAnthropic") as mock_cls, patch(
        "anthropic.types.TextBlock", new=SimpleNamespace
    ):
        fake_client = MagicMock()
        fake_client.messages.create = AsyncMock(side_effect=fake_create)
        mock_cls.return_value = fake_client

        chunks = [_make_chunk(0, content="alpha", context="")]
        cfg = ContextualChunkConfig(enabled=True, lm_client=_make_client("anthropic"))
        await contextualize_chunks_with_llm(chunks, document_text="d", config=cfg)

    assert chunks[0].contextualized == "BLOB\n\nalpha"


# ---------------------------------------------------------------------------
# IngestionService wiring — on/off behavior at the service level
# ---------------------------------------------------------------------------


def _service_chunker(contents: list[str]) -> MagicMock:
    chunker = MagicMock()
    chunker.chunk = MagicMock(
        return_value=[
            ChunkedContent(content=c, page_number=1, chunk_index=i) for i, c in enumerate(contents)
        ]
    )
    return chunker


def _ingest_method() -> SimpleNamespace:
    return SimpleNamespace(name="vector", required=True, ingest=AsyncMock(), delete=AsyncMock())


async def test_ingestion_service_skips_llm_when_disabled() -> None:
    from rfnry_rag.ingestion.chunk.service import IngestionService

    method = _ingest_method()
    service = IngestionService(
        chunker=_service_chunker(["alpha", "beta"]),
        ingestion_methods=[method],
        contextual_chunk=ContextualChunkConfig(enabled=False),
    )

    with patch("anthropic.AsyncAnthropic") as mock_cls:
        await service.ingest_text(content="DOCBODY", metadata={"name": "src"})

    mock_cls.assert_not_called()
    chunks = method.ingest.call_args.kwargs["chunks"]
    assert all(c.situating_context == "" for c in chunks)


async def test_ingestion_service_invokes_llm_when_enabled() -> None:
    from rfnry_rag.ingestion.chunk.service import IngestionService

    captured_docs: list[str] = []

    async def fake_create(**kwargs: object) -> object:
        system = kwargs["system"]
        captured_docs.append(system[0]["text"])  # type: ignore[index]
        return SimpleNamespace(content=[SimpleNamespace(text="SITUATED")])

    method = _ingest_method()
    service = IngestionService(
        chunker=_service_chunker(["alpha", "beta"]),
        ingestion_methods=[method],
        contextual_chunk=ContextualChunkConfig(
            enabled=True,
            lm_client=_make_client("anthropic"),
            concurrency=1,
        ),
    )

    with patch("anthropic.AsyncAnthropic") as mock_cls, patch(
        "anthropic.types.TextBlock", new=SimpleNamespace
    ):
        fake_client = MagicMock()
        fake_client.messages.create = AsyncMock(side_effect=fake_create)
        mock_cls.return_value = fake_client

        await service.ingest_text(content="DOCBODY", metadata={"name": "src"})

    assert len(captured_docs) == 2
    assert all("DOCBODY" in d for d in captured_docs)
    chunks = method.ingest.call_args.kwargs["chunks"]
    assert all(c.situating_context == "SITUATED" for c in chunks)
    assert all("SITUATED" in c.contextualized for c in chunks)
