from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from rfnry_knowledge.providers.usage import TokenUsage

if TYPE_CHECKING:
    from rfnry_knowledge.models import RetrievedChunk, SparseVector


@dataclass(frozen=True)
class EmbeddingResult:
    vectors: list[list[float]]
    usage: TokenUsage | None = None


@dataclass(frozen=True)
class RerankResult:
    chunks: list[RetrievedChunk]
    usage: TokenUsage | None = None


@runtime_checkable
class BaseEmbeddings(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def model(self) -> str: ...

    async def embed(self, texts: list[str]) -> EmbeddingResult: ...

    async def embedding_dimension(self) -> int: ...


@runtime_checkable
class BaseSparseEmbeddings(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def model(self) -> str: ...

    async def embed_sparse(self, texts: list[str]) -> list[SparseVector]: ...

    async def embed_sparse_query(self, query: str) -> SparseVector: ...


@runtime_checkable
class BaseReranking(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def model(self) -> str: ...

    async def rerank(
        self,
        query: str,
        results: list[RetrievedChunk],
        top_k: int = 5,
    ) -> RerankResult: ...


@runtime_checkable
class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...
