from typing import Protocol

from rfnry_rag.retrieval.common.models import RetrievedChunk


class BaseChunkRefinement(Protocol):
    """Protocol for post-retrieval chunk refinement.

    Refiners take a query and retrieved chunks, and return refined chunks.
    Same shape in and out — chunk metadata is preserved.
    """

    async def refine(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]: ...
