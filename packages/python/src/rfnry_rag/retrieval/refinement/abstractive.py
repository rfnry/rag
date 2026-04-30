from dataclasses import replace

from baml_py import errors as baml_errors

from rfnry_rag.baml.baml_client.async_client import b
from rfnry_rag.common.logging import get_logger
from rfnry_rag.models import RetrievedChunk
from rfnry_rag.providers import LanguageModelClient, build_registry
from rfnry_rag.retrieval.common.formatting import format_chunk_header

logger = get_logger("retrieval/refinement/abstractive")


class AbstractiveRefinement:
    """LLM-based abstractive context compression via BAML CompressRetrievedContext.

    Sends all retrieved chunks to an LLM with the query, receives a compressed
    version containing only query-relevant information. The compressed text is
    placed into the first chunk (preserving its metadata); remaining chunks are
    dropped since the LLM has already fused and compressed all content.
    """

    def __init__(self, lm_client: LanguageModelClient, max_output_tokens: int = 1024) -> None:
        self._lm_client = lm_client
        self._max_output_tokens = max_output_tokens

    async def refine(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not chunks:
            return []

        passage_parts = [f"[{i}] {format_chunk_header(chunk)}\n{chunk.content}" for i, chunk in enumerate(chunks)]
        passages = "\n\n".join(passage_parts)

        registry = build_registry(self._lm_client)

        try:
            result = await b.CompressRetrievedContext(
                query=query,
                passages=passages,
                baml_options={"client_registry": registry},
            )

            compressed = result.compressed_text.strip()
            if not compressed:
                logger.warning("abstractive refinement returned empty — keeping original chunks")
                return chunks

            logger.info(
                "abstractive refinement: %d chars -> %d chars (%.0f%% reduction)",
                len(passages),
                len(compressed),
                (1 - len(compressed) / len(passages)) * 100 if passages else 0,
            )

            return [replace(chunks[0], content=compressed)]

        except baml_errors.BamlValidationError as exc:
            logger.exception(
                "CompressRetrievedContext failed: LLM returned unparseable response — "
                "keeping original chunks. Detail: %s",
                exc,
            )
            return chunks

        except Exception as exc:
            logger.exception("CompressRetrievedContext failed: %s — keeping original chunks", exc)
            return chunks
