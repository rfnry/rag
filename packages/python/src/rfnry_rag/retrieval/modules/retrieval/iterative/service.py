"""IterativeRetrievalService — sibling to RetrievalService for multi-hop queries.

R6.1 ships an empty stub. The hop loop, dedup, decomposer wiring, and trace
population are added in R6.2; the engine arm and post-loop DIRECT escalation
land in R6.3. The stub keeps the public name importable so consumers and the
R6.2 implementation can fill it in without churning import sites.
"""

from __future__ import annotations

from typing import Any


class IterativeRetrievalService:
    """Sibling to ``RetrievalService`` for multi-hop iterative queries.

    R6.1 ships only the public name. ``retrieve`` raises
    ``NotImplementedError`` until R6.2 lands the hop loop.
    """

    async def retrieve(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("IterativeRetrievalService lands in R6.2")
