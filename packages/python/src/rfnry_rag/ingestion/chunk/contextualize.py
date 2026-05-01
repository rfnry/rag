"""LLM-driven situating context per chunk (Anthropic Contextual Retrieval).

Sibling of ``expand.py``. Native SDK dispatch (no BAML); per-provider blocks
mirror ``providers/text_generation.py``. The document body is sent as a
stable cached prefix per provider so subsequent per-chunk calls hit the
prompt cache.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rfnry_rag.concurrency import run_concurrent
from rfnry_rag.exceptions import ConfigurationError, EnrichmentSkipped, IngestionError
from rfnry_rag.ingestion.chunk.token_counter import count_tokens
from rfnry_rag.ingestion.models import ChunkedContent
from rfnry_rag.logging import get_logger

if TYPE_CHECKING:
    from rfnry_rag.config.ingestion import ContextualChunkConfig
    from rfnry_rag.providers.provider import LanguageModel

logger = get_logger("ingestion.chunk.contextualize")


_PROMPT_TEMPLATE = (
    "Here is the chunk we want to situate within the whole document:\n"
    "<chunk>\n{chunk}\n</chunk>\n"
    "Please give a short succinct context (no more than {max_tokens} tokens) to "
    "situate this chunk within the overall document for the purposes of improving "
    "search retrieval of the chunk. Answer only with the succinct context and "
    "nothing else."
)

DOC_RESERVE_TOKENS = 16_000
DEFAULT_DOC_CAP = 150_000


async def contextualize_chunks_with_llm(
    chunks: list[ChunkedContent],
    document_text: str,
    config: ContextualChunkConfig,
) -> list[ChunkedContent]:
    """Attach LLM-generated situating context to each chunk.

    Mutates each chunk's ``situating_context`` field in-place and refreshes
    ``contextualized`` to fold it in alongside the structural header
    (already on ``chunk.context``) and the original content. No-ops when
    ``config.enabled`` is False or ``chunks`` is empty. Per-chunk failure
    raises ``IngestionError`` carrying the chunk_index.
    """
    if not config.enabled or not chunks:
        return chunks
    client = config.lm_client
    assert client is not None  # __post_init__ guarantees this when enabled

    window = client.lm.context_size or DEFAULT_DOC_CAP
    cap = window - DOC_RESERVE_TOKENS - config.max_context_tokens
    doc_tokens = count_tokens(document_text)
    if doc_tokens > cap:
        raise EnrichmentSkipped(
            "contextual_chunk",
            f"document_too_large({doc_tokens}>{cap}_cap)",
        )

    async def _situate_one(chunk: ChunkedContent) -> None:
        try:
            blob = await _generate_situating_context(
                lm=client.lm,
                document=document_text,
                chunk=chunk.content,
                max_tokens=config.max_context_tokens,
                max_retries=client.max_retries,
                timeout_seconds=client.timeout_seconds,
                temperature=client.temperature,
            )
        except Exception as exc:
            raise IngestionError(
                f"chunk contextualization failed for chunk_index={chunk.chunk_index}: {exc}"
            ) from exc
        chunk.situating_context = blob
        chunk.contextualized = _fold(chunk)

    await run_concurrent(chunks, _situate_one, concurrency=config.concurrency)
    logger.info(
        "contextualized %d chunks with situating context (concurrency=%d)",
        len(chunks),
        config.concurrency,
    )
    return chunks


def _fold(chunk: ChunkedContent) -> str:
    parts = [p for p in (chunk.context, chunk.situating_context, chunk.content) if p]
    return "\n\n".join(parts)


async def _generate_situating_context(
    *,
    lm: LanguageModel,
    document: str,
    chunk: str,
    max_tokens: int,
    max_retries: int,
    timeout_seconds: int,
    temperature: float,
) -> str:
    if lm.provider == "anthropic":
        return await _anthropic_situate(
            lm=lm,
            document=document,
            chunk=chunk,
            max_tokens=max_tokens,
            max_retries=max_retries,
            timeout_seconds=timeout_seconds,
            temperature=temperature,
        )
    if lm.provider == "openai":
        return await _openai_situate(
            lm=lm,
            document=document,
            chunk=chunk,
            max_tokens=max_tokens,
            max_retries=max_retries,
            timeout_seconds=timeout_seconds,
            temperature=temperature,
        )
    if lm.provider == "gemini":
        return await _gemini_situate(
            lm=lm,
            document=document,
            chunk=chunk,
            max_tokens=max_tokens,
            max_retries=max_retries,
            timeout_seconds=timeout_seconds,
            temperature=temperature,
        )
    raise ConfigurationError(
        f"Unsupported provider for ContextualChunkConfig: {lm.provider!r}. "
        f"Supported: anthropic, openai, gemini."
    )


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


async def _anthropic_situate(
    *,
    lm: LanguageModel,
    document: str,
    chunk: str,
    max_tokens: int,
    max_retries: int,
    timeout_seconds: int,
    temperature: float,
) -> str:
    from anthropic import AsyncAnthropic
    from anthropic.types import TextBlock

    client = AsyncAnthropic(api_key=lm.api_key, max_retries=max_retries, timeout=timeout_seconds)
    user_message = _PROMPT_TEMPLATE.format(chunk=chunk, max_tokens=max_tokens)
    # The document body lives on the system parameter as a list-of-blocks
    # so cache_control marks it as the stable prefix shared across chunks.
    response = await client.messages.create(
        model=lm.model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=[
            {
                "type": "text",
                "text": f"<document>\n{document}\n</document>",
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_message}],
    )
    first_block = response.content[0] if response.content else None
    return first_block.text if isinstance(first_block, TextBlock) else ""


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------


async def _openai_situate(
    *,
    lm: LanguageModel,
    document: str,
    chunk: str,
    max_tokens: int,
    max_retries: int,
    timeout_seconds: int,
    temperature: float,
) -> str:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=lm.api_key, max_retries=max_retries, timeout=timeout_seconds)
    user_message = _PROMPT_TEMPLATE.format(chunk=chunk, max_tokens=max_tokens)
    # Stable system prefix → OpenAI's automatic prefix cache applies once
    # the request crosses the provider's size threshold; no explicit knob.
    response = await client.chat.completions.create(
        model=lm.model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[
            {"role": "system", "content": f"<document>\n{document}\n</document>"},
            {"role": "user", "content": user_message},
        ],
    )
    content = response.choices[0].message.content
    return content or ""


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------


async def _gemini_situate(
    *,
    lm: LanguageModel,
    document: str,
    chunk: str,
    max_tokens: int,
    max_retries: int,
    timeout_seconds: int,
    temperature: float,
) -> str:
    from google import genai
    from google.genai import types

    http_options = types.HttpOptions(
        timeout=timeout_seconds * 1000,
        retry_options=types.HttpRetryOptions(attempts=max_retries),
    )
    client = genai.Client(api_key=lm.api_key, http_options=http_options)
    user_message = _PROMPT_TEMPLATE.format(chunk=chunk, max_tokens=max_tokens)
    response = await client.aio.models.generate_content(
        model=lm.model,
        contents=user_message,
        config=types.GenerateContentConfig(
            system_instruction=f"<document>\n{document}\n</document>",
            max_output_tokens=max_tokens,
            temperature=temperature,
        ),
    )
    return response.text or ""
