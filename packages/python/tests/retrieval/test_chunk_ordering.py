import pytest

from rfnry_rag.generation.formatting import ChunkOrdering, chunks_to_context
from rfnry_rag.models import RetrievedChunk


def _make_chunk(index: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"chunk_{index}",
        source_id="s1",
        content=f"chunk_{index}",
        score=1.0 - index * 0.1,
        page_number=index + 1,
        source_metadata={"name": "doc"},
    )


def _positions(context: str, chunks: list[RetrievedChunk]) -> list[int]:
    return [context.index(c.content) for c in chunks]


def test_chunks_to_context_score_descending_is_identity() -> None:
    chunks = [_make_chunk(i) for i in range(5)]
    context = chunks_to_context(chunks, ordering=ChunkOrdering.SCORE_DESCENDING)
    positions = _positions(context, chunks)
    assert positions == sorted(positions)


def test_chunks_to_context_primacy_recency_reorders_to_edges() -> None:
    chunks = [_make_chunk(i) for i in range(5)]
    context = chunks_to_context(chunks, ordering=ChunkOrdering.PRIMACY_RECENCY)
    pos = {c.content: context.index(c.content) for c in chunks}
    assert pos["chunk_0"] < pos["chunk_2"] < pos["chunk_4"] < pos["chunk_3"] < pos["chunk_1"]


def test_chunks_to_context_sandwich_reorders_top_then_reversed() -> None:
    chunks = [_make_chunk(i) for i in range(5)]
    context = chunks_to_context(chunks, ordering=ChunkOrdering.SANDWICH)
    pos = {c.content: context.index(c.content) for c in chunks}
    assert pos["chunk_0"] < pos["chunk_1"] < pos["chunk_4"] < pos["chunk_3"] < pos["chunk_2"]


@pytest.mark.parametrize(
    "ordering",
    [ChunkOrdering.SCORE_DESCENDING, ChunkOrdering.PRIMACY_RECENCY, ChunkOrdering.SANDWICH],
)
@pytest.mark.parametrize("size", [1, 2])
def test_chunks_to_context_short_inputs_are_stable(ordering: ChunkOrdering, size: int) -> None:
    chunks = [_make_chunk(i) for i in range(size)]
    context = chunks_to_context(chunks, ordering=ordering)
    positions = _positions(context, chunks)
    assert positions == sorted(positions)
