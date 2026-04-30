from __future__ import annotations

from typing import Protocol


class BaseEmbeddings(Protocol):
    @property
    def model(self) -> str: ...

    @property
    def name(self) -> str: ...

    async def embed(self, texts: list[str]) -> list[list[float]]: ...

    async def embedding_dimension(self) -> int: ...
