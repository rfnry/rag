from typing import Protocol

from rfnry_knowledge.models import RetrievedChunk


class BaseReranking(Protocol):
    async def rerank(self, query: str, results: list[RetrievedChunk], top_k: int = 5) -> list[RetrievedChunk]: ...
