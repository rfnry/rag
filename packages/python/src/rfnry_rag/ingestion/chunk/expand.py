"""LLM-driven document expansion: synthetic queries per chunk.

Sibling to ``chunk/context.py``. Kept separate because expansion is async +
LLM-driven; contextualisation is pure string templating.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rfnry_rag.baml.baml_client.async_client import b
from rfnry_rag.concurrency import run_concurrent
from rfnry_rag.ingestion.models import ChunkedContent
from rfnry_rag.ingestion.notes import record_skip
from rfnry_rag.logging import get_logger
from rfnry_rag.telemetry.usage import instrument_baml_call

if TYPE_CHECKING:
    from baml_py import ClientRegistry

    from rfnry_rag.config.ingestion import DocumentExpansionConfig

logger = get_logger("ingestion.chunk.expand")

_MAJORITY_FAILURE_RATIO = 0.5


async def expand_chunks(
    chunks: list[ChunkedContent],
    config: DocumentExpansionConfig,
    registry: ClientRegistry,
    notes: list[str] | None = None,
) -> list[ChunkedContent]:
    """Attach synthetic queries to each chunk via the configured LLM.

    Per-chunk failures soft-skip: the chunk ends with an empty
    ``synthetic_queries`` list and a ``document_expansion:warn:chunk_<i>:...``
    note is appended to the caller-supplied list. When more than half of the
    chunks fail, an additional ``document_expansion:warn:majority_failed(f/t)``
    summary note records the overall feature degradation.

    Bias-term hygiene: this helper never inspects chunk content textually; the
    BAML prompt body is the only consumer-facing surface, audited by the
    domain-agnostic contract test.
    """
    if not config.enabled or not chunks:
        return chunks

    failed_count = 0

    async def _expand_one(chunk: ChunkedContent) -> None:
        nonlocal failed_count
        try:
            result = await instrument_baml_call(
                operation="document_expansion",
                call=lambda collector: b.GenerateSyntheticQueries(
                    chunk.content,
                    config.num_queries,
                    baml_options={"client_registry": registry, "collector": collector},
                ),
            )
        except Exception as exc:
            failed_count += 1
            await record_skip(
                notes,
                step="document_expansion",
                level="warn",
                reason=f"chunk_{chunk.chunk_index}:failed({exc!s:.80})",
            )
            chunk.synthetic_queries = []
            return
        chunk.synthetic_queries = list(result.queries)

    await run_concurrent(chunks, _expand_one, concurrency=config.concurrency)

    total = len(chunks)
    if total > 0 and failed_count / total > _MAJORITY_FAILURE_RATIO:
        await record_skip(
            notes,
            step="document_expansion",
            level="warn",
            reason=f"majority_failed({failed_count}/{total})",
        )

    logger.info(
        "expanded %d chunks with synthetic queries (concurrency=%d, failed=%d)",
        total,
        config.concurrency,
        failed_count,
    )
    return chunks
