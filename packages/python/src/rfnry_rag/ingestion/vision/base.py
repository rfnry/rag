from typing import Protocol

from rfnry_rag.ingestion.models import ParsedPage


class BaseVision(Protocol):
    async def parse(self, file_path: str, pages: set[int] | None = None) -> list[ParsedPage]: ...
