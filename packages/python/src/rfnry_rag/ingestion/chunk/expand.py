"""LLM-driven document expansion: synthetic queries per chunk.

Sibling to ``chunk/context.py``. Kept separate because expansion is async +
LLM-driven; contextualisation is pure string templating.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rfnry_rag.baml.baml_client.async_client import b
from rfnry_rag.common.concurrency import run_concurrent
from rfnry_rag.common.logging import get_logger
from rfnry_rag.exceptions import IngestionError
from rfnry_rag.ingestion.models import ChunkedContent

if TYPE_CHECKING:
    from baml_py import ClientRegistry

    from rfnry_rag.server import DocumentExpansionConfig

logger = get_logger("ingestion.chunk.expand")


async def expand_chunks(
    chunks: list[ChunkedContent],
    config: DocumentExpansionConfig,
    registry: ClientRegistry,
) -> list[ChunkedContent]:
    """Attach synthetic queries to each chunk via the configured LLM.

    Mutates each chunk's ``synthetic_queries`` field in-place and returns the
    same list for caller convenience. No-ops when ``config.enabled`` is False
    or the chunk list is empty.

    Bias-term hygiene: this helper never inspects chunk content textually; the
    BAML prompt body is the only consumer-facing surface, audited by the
    domain-agnostic contract test.
    """
    if not config.enabled or not chunks:
        return chunks

    async def _expand_one(chunk: ChunkedContent) -> None:
        try:
            result = await b.GenerateSyntheticQueries(
                chunk.content,
                config.num_queries,
                baml_options={"client_registry": registry},
            )
        except Exception as exc:
            raise IngestionError(f"document expansion failed for chunk_index={chunk.chunk_index}: {exc}") from exc
        chunk.synthetic_queries = list(result.queries)

    await run_concurrent(chunks, _expand_one, concurrency=config.concurrency)
    logger.info("expanded %d chunks with synthetic queries (concurrency=%d)", len(chunks), config.concurrency)
    return chunks
