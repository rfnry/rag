from __future__ import annotations

from typing import TYPE_CHECKING

from rfnry_knowledge.common.concurrency import run_concurrent
from rfnry_knowledge.common.logging import get_logger
from rfnry_knowledge.exceptions import ConfigurationError, EnrichmentSkipped, IngestionError
from rfnry_knowledge.ingestion.chunk.token_counter import count_tokens
from rfnry_knowledge.ingestion.models import ChunkedContent
from rfnry_knowledge.providers.provider import (
    AnthropicModelProvider,
    GoogleModelProvider,
    ModelProvider,
    OpenAIModelProvider,
)
from rfnry_knowledge.telemetry.context import increment_ingest_field
from rfnry_knowledge.telemetry.usage import (
    extract_anthropic_usage,
    extract_gemini_usage,
    extract_openai_usage,
    instrument_call,
)

if TYPE_CHECKING:
    from rfnry_knowledge.config.ingestion import ContextualChunkConfig

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
    if not config.enabled or not chunks:
        return chunks
    client = config.lm_client
    assert client is not None

    window = client.provider.context_size or DEFAULT_DOC_CAP
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
                provider=client.provider,
                document=document_text,
                chunk=chunk.content,
                max_tokens=config.max_context_tokens,
                max_retries=client.max_retries,
                timeout_seconds=client.timeout_seconds,
                temperature=client.temperature,
            )
        except Exception as exc:
            raise IngestionError(f"chunk contextualization failed for chunk_index={chunk.chunk_index}: {exc}") from exc
        chunk.situating_context = blob
        chunk.contextualized = _fold(chunk)
        increment_ingest_field("contextual_chunk_calls")

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
    provider: ModelProvider,
    document: str,
    chunk: str,
    max_tokens: int,
    max_retries: int,
    timeout_seconds: int,
    temperature: float,
) -> str:
    if isinstance(provider, AnthropicModelProvider):
        return await _anthropic_situate(
            provider=provider,
            document=document,
            chunk=chunk,
            max_tokens=max_tokens,
            max_retries=max_retries,
            timeout_seconds=timeout_seconds,
            temperature=temperature,
        )
    if isinstance(provider, OpenAIModelProvider):
        return await _openai_situate(
            provider=provider,
            document=document,
            chunk=chunk,
            max_tokens=max_tokens,
            max_retries=max_retries,
            timeout_seconds=timeout_seconds,
            temperature=temperature,
        )
    if isinstance(provider, GoogleModelProvider):
        return await _google_situate(
            provider=provider,
            document=document,
            chunk=chunk,
            max_tokens=max_tokens,
            max_retries=max_retries,
            timeout_seconds=timeout_seconds,
            temperature=temperature,
        )
    raise ConfigurationError(
        f"Unsupported provider for ContextualChunkConfig: {provider.kind!r}. Supported: anthropic, openai, google."
    )


async def _anthropic_situate(
    *,
    provider: AnthropicModelProvider,
    document: str,
    chunk: str,
    max_tokens: int,
    max_retries: int,
    timeout_seconds: int,
    temperature: float,
) -> str:
    from anthropic import AsyncAnthropic
    from anthropic.types import TextBlock

    client = AsyncAnthropic(
        api_key=provider.api_key.get_secret_value(),
        base_url=provider.base_url,
        max_retries=max_retries,
        timeout=timeout_seconds,
    )
    user_message = _PROMPT_TEMPLATE.format(chunk=chunk, max_tokens=max_tokens)
    response = await instrument_call(
        provider="anthropic",
        model=provider.model,
        operation="contextualize",
        extract_usage=extract_anthropic_usage,
        call=lambda: client.messages.create(
            model=provider.model,
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
        ),
    )
    first_block = response.content[0] if response.content else None
    return first_block.text if isinstance(first_block, TextBlock) else ""


async def _openai_situate(
    *,
    provider: OpenAIModelProvider,
    document: str,
    chunk: str,
    max_tokens: int,
    max_retries: int,
    timeout_seconds: int,
    temperature: float,
) -> str:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=provider.api_key.get_secret_value(),
        base_url=provider.base_url,
        organization=provider.organization,
        project=provider.project,
        max_retries=max_retries,
        timeout=timeout_seconds,
    )
    user_message = _PROMPT_TEMPLATE.format(chunk=chunk, max_tokens=max_tokens)
    response = await instrument_call(
        provider="openai",
        model=provider.model,
        operation="contextualize",
        extract_usage=extract_openai_usage,
        call=lambda: client.chat.completions.create(
            model=provider.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": f"<document>\n{document}\n</document>"},
                {"role": "user", "content": user_message},
            ],
        ),
    )
    content = response.choices[0].message.content
    return content or ""


async def _google_situate(
    *,
    provider: GoogleModelProvider,
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
    client = genai.Client(api_key=provider.api_key.get_secret_value(), http_options=http_options)
    user_message = _PROMPT_TEMPLATE.format(chunk=chunk, max_tokens=max_tokens)
    response = await instrument_call(
        provider="google",
        model=provider.model,
        operation="contextualize",
        extract_usage=extract_gemini_usage,
        call=lambda: client.aio.models.generate_content(
            model=provider.model,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=f"<document>\n{document}\n</document>",
                max_output_tokens=max_tokens,
                temperature=temperature,
            ),
        ),
    )
    return response.text or ""
