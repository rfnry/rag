from enum import Enum

from rfnry_knowledge.exceptions import ConfigurationError
from rfnry_knowledge.models import RetrievedChunk


class ChunkOrdering(Enum):
    SCORE_DESCENDING = "score_descending"
    PRIMACY_RECENCY = "primacy_recency"
    SANDWICH = "sandwich"


def format_chunk_header(chunk: RetrievedChunk) -> str:
    """Format a chunk's source attribution header."""
    source_name = chunk.source_metadata.get("name", "")
    page_ref = f"Page {chunk.page_number}" if chunk.page_number else "Unknown page"
    section = f" - {chunk.section}" if chunk.section else ""
    if source_name:
        return f"[{source_name} — {page_ref}{section}]"
    return f"[{page_ref}{section}]"


def _reorder_primacy_recency(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    evens = chunks[0::2]
    odds = chunks[1::2]
    return evens + list(reversed(odds))


def _reorder_sandwich(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    top_two = chunks[:2]
    tail_reversed = chunks[2:][::-1]
    return top_two + tail_reversed


def chunks_to_context(
    chunks: list[RetrievedChunk],
    *,
    ordering: ChunkOrdering = ChunkOrdering.SCORE_DESCENDING,
) -> str:
    """Format retrieved chunks into a context string with source attribution headers.

    Liu et al. (TACL 2024, "Lost in the Middle") show U-shaped attention in
    long-context LLMs: tokens at the start and end of the prompt are attended
    to more than the middle. The retrieval pipeline emits chunks in
    score-descending order, which puts the second-best chunk in the
    attention-poor middle. ``ordering`` lets callers opt into edge-biased
    layouts that move the best chunks to the recency-privileged tail.
    """
    if ordering is ChunkOrdering.SCORE_DESCENDING:
        ordered = chunks
    elif ordering is ChunkOrdering.PRIMACY_RECENCY:
        ordered = _reorder_primacy_recency(chunks)
    elif ordering is ChunkOrdering.SANDWICH:
        ordered = _reorder_sandwich(chunks)
    else:
        raise ConfigurationError(f"unknown chunk ordering: {ordering!r}")

    parts = [f"{format_chunk_header(chunk)}\n{chunk.content}" for chunk in ordered]
    return "\n\n---\n\n".join(parts)
