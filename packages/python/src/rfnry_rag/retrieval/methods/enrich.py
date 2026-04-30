"""`BaseRetrievalMethod` adapter for structured (enrich) retrieval.

Wraps `StructuredRetrievalService.retrieve` so it can be composed into a
`RetrievalService` as a plug-compatible `BaseRetrievalMethod`.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from rfnry_rag.logging import get_logger
from rfnry_rag.models import RetrievedChunk

if TYPE_CHECKING:
    from rfnry_rag.retrieval.enrich.service import StructuredRetrievalService

logger = get_logger("retrieval.methods.enrich")


class StructuredRetrieval:
    """Adapts `StructuredRetrievalService` to the `BaseRetrievalMethod` protocol.

    The ``.name`` is ``"enrich"`` (not ``"structured"``) for historical reasons —
    the method was originally called the "enrich" step before it was reframed as
    a retrieval method in its own right. Dispatch sites and logs use that name.
    """

    def __init__(
        self,
        service: StructuredRetrievalService,
        weight: float = 1.0,
        top_k: int | None = None,
    ) -> None:
        self._service = service
        self._weight = weight
        self._top_k = top_k

    @property
    def name(self) -> str:
        return "enrich"

    @property
    def weight(self) -> float:
        return self._weight

    @property
    def top_k(self) -> int | None:
        return self._top_k

    async def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        knowledge_id: str | None = None,
    ) -> list[RetrievedChunk]:
        start = time.perf_counter()
        try:
            effective_top_k = self._top_k if self._top_k is not None else top_k
            results = await self._service.retrieve(query=query, knowledge_id=knowledge_id, top_k=effective_top_k)
            elapsed = (time.perf_counter() - start) * 1000
            logger.info("%d results in %.1fms", len(results), elapsed)
            return results
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000
            logger.warning("failed in %.1fms — %s", elapsed, exc)
            return []
