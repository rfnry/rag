"""RaptorRetrieval — query-time search over RAPTOR summary vectors.

``RaptorRetrieval`` is a sibling ``BaseRetrievalMethod`` (alongside
``VectorRetrieval`` / ``DocumentRetrieval`` / ``GraphRetrieval``) that
fetches the active tree id from the registry, embeds the query, and runs
a payload-filtered vector search restricted to that tree's summaries.
``RaptorTreeBuilder`` (via ``RagEngine.build_raptor_index``) writes the
summary vectors tagged ``vector_role="raptor_summary"`` and stamps the
active tree pointer on the registry. Results land in the same RRF
fusion pool as the other methods — no special-casing downstream.

Concrete behaviour:
- ``knowledge_id is None`` short-circuits to ``[]``. RAPTOR is a per-
  ``knowledge_id`` strategy by design (one tree per scope; cross-scope
  global trees are explicitly out of scope). Cross-scope queries proceed
  through vector / document / graph retrieval without a RAPTOR
  contribution.
- No active tree (registry returns ``None``) → ``[]``. Retrieval continues
  with the other methods; RAPTOR is additive, not gating.
- Summaries span sources by design; ``RetrievedChunk.source_id`` is set to
  the sentinel ``f"raptor:{tree_id}"`` rather than ``None`` because the
  dataclass field is non-optional and downstream consumers may group/dedupe
  by ``source_id`` (a ``None`` value would degrade those paths).
"""

from __future__ import annotations

import time
from typing import Any

from rfnry_rag.retrieval.common.logging import get_logger
from rfnry_rag.retrieval.common.models import RetrievedChunk, VectorResult
from rfnry_rag.retrieval.modules.ingestion.embeddings.base import BaseEmbeddings
from rfnry_rag.retrieval.modules.ingestion.methods.raptor.registry import (
    RaptorTreeRegistry,
)
from rfnry_rag.retrieval.stores.vector.base import BaseVectorStore

logger = get_logger("retrieval.methods.raptor")


class RaptorRetrieval:
    """RAPTOR-summary retrieval: dense search over the active tree's summaries.

    Sibling to ``VectorRetrieval`` — it does NOT layer on top of it. The
    ``vector_role="raptor_summary"`` payload filter is what isolates summary
    vectors from leaf vectors so the same Qdrant collection holds both
    without crosstalk. The ``raptor_tree_id`` filter scopes to the
    currently-active tree so a stale orphan tree (left over after a build
    swap) cannot leak into results even if GC has not yet run.
    """

    def __init__(
        self,
        vector_store: BaseVectorStore,
        embeddings: BaseEmbeddings,
        registry: RaptorTreeRegistry,
        weight: float = 1.0,
        top_k: int | None = None,
    ) -> None:
        self._store = vector_store
        self._embeddings = embeddings
        self._registry = registry
        self._weight = weight
        self._top_k = top_k

    @property
    def name(self) -> str:
        return "raptor"

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
        # RAPTOR is per-knowledge_id by design: cross-scope global trees
        # are explicitly out of scope. A query without a knowledge_id
        # cannot identify which tree to consult, so we contribute nothing
        # and let the other methods drive cross-scope retrieval.
        if knowledge_id is None:
            logger.debug("no knowledge_id — RAPTOR returns []")
            return []

        try:
            active_tree_id = await self._registry.get_active(knowledge_id)
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000
            logger.warning("registry lookup failed in %.1fms — %s", elapsed, exc)
            return []

        if active_tree_id is None:
            # Tree not built yet (or was wiped by knowledge removal). RRF
            # fusion proceeds without a RAPTOR contribution; no error.
            logger.debug("no active tree for knowledge_id=%s — RAPTOR returns []", knowledge_id)
            return []

        try:
            results = await self._do_search(query, top_k, knowledge_id, active_tree_id)
            elapsed = (time.perf_counter() - start) * 1000
            logger.info("%d results in %.1fms", len(results), elapsed)
            return results
        except Exception:
            elapsed = (time.perf_counter() - start) * 1000
            logger.warning("raptor retrieval failed in %.1fms", elapsed, exc_info=True)
            return []

    async def _do_search(
        self,
        query: str,
        top_k: int,
        knowledge_id: str,
        active_tree_id: str,
    ) -> list[RetrievedChunk]:
        vectors = await self._embeddings.embed([query])
        if not vectors:
            logger.warning("embedding returned no vectors for query")
            return []
        query_vector = vectors[0]

        # vector_role="raptor_summary" implies raptor_level >= 1 because
        # leaves carry roles {description, raw_text, table_row}; the tree_id
        # filter scopes to the active build so orphans from prior builds
        # cannot bleed in even if GC has not run yet.
        payload_filter: dict[str, Any] = {
            "knowledge_id": knowledge_id,
            "vector_role": "raptor_summary",
            "raptor_tree_id": active_tree_id,
        }

        raw_results = await self._store.search(
            vector=query_vector,
            top_k=top_k,
            filters=payload_filter,
        )
        return [self._result_to_chunk(r, active_tree_id) for r in raw_results]

    @staticmethod
    def _result_to_chunk(result: VectorResult, active_tree_id: str) -> RetrievedChunk:
        payload = result.payload
        # Sentinel rather than None: ``RetrievedChunk.source_id`` is a non-
        # optional ``str`` field and downstream consumers (e.g. dedup,
        # source-type weighting) may key on it. The sentinel keeps those
        # paths well-formed while signalling "not a single source".
        sentinel_source_id = f"raptor:{active_tree_id}"
        content = payload.get("content") or payload.get("contextualized") or ""
        return RetrievedChunk(
            chunk_id=str(result.point_id),
            source_id=sentinel_source_id,
            content=content,
            score=result.score,
            metadata={
                "raptor_tree_id": payload.get("raptor_tree_id"),
                "raptor_level": payload.get("raptor_level"),
                "raptor_cluster_size": payload.get("raptor_cluster_size"),
                "raptor_parent_id": payload.get("raptor_parent_id"),
                "raptor_reasoning": payload.get("raptor_reasoning"),
            },
            source_metadata={
                "retrieval_type": "raptor",
            },
        )
