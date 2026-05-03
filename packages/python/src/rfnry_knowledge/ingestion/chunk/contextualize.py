from __future__ import annotations

from typing import TYPE_CHECKING

from rfnry_knowledge.common.concurrency import run_concurrent
from rfnry_knowledge.common.logging import get_logger
from rfnry_knowledge.exceptions import ConfigurationError, EnrichmentSkipped, IngestionError
from rfnry_knowledge.ingestion.models import ChunkedContent
from rfnry_knowledge.providers.provider import ProviderClient
from rfnry_knowledge.providers.registry import build_registry
from rfnry_knowledge.telemetry.context import increment_ingest_field
from rfnry_knowledge.telemetry.usage import instrument_baml_call

if TYPE_CHECKING:
    from rfnry_knowledge.config.ingestion import ContextualChunkConfig

logger = get_logger("ingestion.chunk.contextualize")


DOC_RESERVE_TOKENS = 16_000
DEFAULT_DOC_CAP = 150_000


async def contextualize_chunks_with_llm(
    chunks: list[ChunkedContent],
    document_text: str,
    config: ContextualChunkConfig,
) -> list[ChunkedContent]:
    if not config.enabled or not chunks:
        return chunks
    client = config.provider_client
    if client is None:
        raise ConfigurationError("ContextualChunkConfig.provider_client is required when enabled=True")
    counter = config.token_counter
    if counter is None:
        raise ConfigurationError("ContextualChunkConfig.token_counter is required when enabled=True")

    window = client.context_size or DEFAULT_DOC_CAP
    cap = window - DOC_RESERVE_TOKENS - config.max_context_tokens
    doc_tokens = counter.count(document_text)
    if doc_tokens > cap:
        raise EnrichmentSkipped(
            "contextual_chunk",
            f"document_too_large({doc_tokens}>{cap}_cap)",
        )

    async def _situate_one(chunk: ChunkedContent) -> None:
        try:
            blob = await _situate_via_baml(
                client=client,
                document=document_text,
                chunk=chunk.content,
                max_tokens=config.max_context_tokens,
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


async def _situate_via_baml(
    *,
    client: ProviderClient,
    document: str,
    chunk: str,
    max_tokens: int,
) -> str:
    from rfnry_knowledge.baml.baml_client.async_client import b

    registry = build_registry(client)

    async def _call(collector):
        return await b.SituateChunk(
            document=document,
            chunk=chunk,
            max_tokens=max_tokens,
            baml_options={"client_registry": registry, "collector": collector},
        )

    response = await instrument_baml_call(
        operation="contextualize",
        call=_call,
        fallback_provider=client.name,
        fallback_model=client.model,
    )
    return response if isinstance(response, str) else str(response)
