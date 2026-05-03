from typing import Protocol

from rfnry_knowledge.models import SparseVector


class BaseSparseEmbeddings(Protocol):
    @property
    def model(self) -> str: ...

    async def embed_sparse(self, texts: list[str]) -> list[SparseVector]: ...

    async def embed_sparse_query(self, query: str) -> SparseVector: ...
