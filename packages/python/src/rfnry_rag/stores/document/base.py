from typing import Protocol

from rfnry_rag.models import ContentMatch


class BaseDocumentStore(Protocol):
    async def initialize(self) -> None: ...

    async def store_content(
        self,
        source_id: str,
        knowledge_id: str | None,
        source_type: str | None,
        title: str,
        content: str,
    ) -> None: ...

    async def search_content(
        self,
        query: str,
        knowledge_id: str | None = None,
        source_type: str | None = None,
        top_k: int = 5,
    ) -> list[ContentMatch]: ...

    async def get(self, source_id: str) -> str | None: ...

    async def delete_content(self, source_id: str) -> None: ...

    async def shutdown(self) -> None: ...
