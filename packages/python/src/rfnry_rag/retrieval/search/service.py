from __future__ import annotations

import asyncio
import time
from typing import Any

from rfnry_rag.logging import get_logger, query_logging_enabled
from rfnry_rag.models import RetrievedChunk
from rfnry_rag.observability.trace import RetrievalTrace
from rfnry_rag.retrieval.base import BaseRetrievalMethod
from rfnry_rag.retrieval.search.fusion import reciprocal_rank_fusion
from rfnry_rag.retrieval.search.reranking.base import BaseReranking

logger = get_logger("retrieval.search.service")


class RetrievalService:
    def __init__(
        self,
        retrieval_methods: list[BaseRetrievalMethod],
        reranking: BaseReranking | None = None,
        top_k: int = 5,
        source_type_weights: dict[str, float] | None = None,
    ) -> None:
        self._retrieval_methods = retrieval_methods
        self._reranking = reranking
        self._top_k = top_k
        self._source_type_weights = source_type_weights

    @property
    def methods(self) -> list[BaseRetrievalMethod]:
        """Public read-only view over configured retrieval methods.

        This is NOT a Protocol member — RetrievalService is the concrete and
        sole service implementation today. If the service ever becomes a swap
        point (e.g., for a streaming variant), lift this property into a
        BaseRetrievalService Protocol first.

        Returns the live list — callers must not mutate.
        """
        return self._retrieval_methods

    async def retrieve(
        self,
        query: str,
        knowledge_id: str | None = None,
        top_k: int | None = None,
        trace: bool = False,
    ) -> tuple[list[RetrievedChunk], RetrievalTrace | None]:
        if not query or not query.strip():
            if trace:
                return [], RetrievalTrace(query=query, knowledge_id=knowledge_id)
            return [], None

        trace_obj: RetrievalTrace | None = RetrievalTrace(query=query, knowledge_id=knowledge_id) if trace else None

        effective_top_k = top_k if top_k is not None else self._top_k
        fetch_k = effective_top_k * 4
        top_k = effective_top_k
        filters = self._build_filters(knowledge_id)

        # User query text is PII-adjacent; log only when explicitly opted in
        # via RFNRY_RAG_LOG_QUERIES=true. Always log the knowledge_id + length.
        if query_logging_enabled():
            logger.info('query: "%s" (knowledge_id=%s)', query[:80], knowledge_id)
        else:
            logger.info("query: (len=%d, knowledge_id=%s)", len(query), knowledge_id)

        retrieval_start = time.perf_counter() if trace_obj is not None else 0.0
        all_result_lists, all_weights, by_method = await self._search_single_query(
            query,
            fetch_k,
            filters,
            knowledge_id,
            collect_per_method=trace_obj is not None,
        )

        if trace_obj is not None:
            per_method: dict[str, list[RetrievedChunk]] = {m.name: [] for m in self._retrieval_methods}
            if by_method is not None:
                for method_name, results in by_method.items():
                    per_method[method_name] = list(results)
            trace_obj.per_method_results = per_method
            trace_obj.timings["retrieval"] = time.perf_counter() - retrieval_start

        fusion_start = time.perf_counter() if trace_obj is not None else 0.0
        if len(all_result_lists) > 1:
            fused = reciprocal_rank_fusion(
                all_result_lists,
                source_type_weights=self._source_type_weights,
                method_weights=all_weights,
            )
            logger.info("%d unique after reciprocal rank fusion", len(fused))
        elif all_result_lists:
            fused = self._apply_source_weights(all_result_lists[0])
        else:
            fused = []

        if trace_obj is not None:
            trace_obj.fused_results = list(fused)
            trace_obj.timings["fusion"] = time.perf_counter() - fusion_start

        if self._reranking:
            reranking_start = time.perf_counter() if trace_obj is not None else 0.0
            if fused:
                fused = await self._reranking.rerank(query, fused, top_k=top_k)
                logger.info("top %d selected after reranking", len(fused))
            if trace_obj is not None:
                # Reranker is configured: trace records the post-reranking
                # state even when the input was empty (an empty list, not None).
                trace_obj.reranked_results = list(fused)
                trace_obj.timings["reranking"] = time.perf_counter() - reranking_start
        else:
            fused = fused[:top_k]

        if trace_obj is not None:
            trace_obj.final_results = list(fused)

        return fused, trace_obj

    async def _search_single_query(
        self,
        query: str,
        fetch_k: int,
        filters: dict[str, Any] | None,
        knowledge_id: str | None,
        collect_per_method: bool = False,
    ) -> tuple[list[list[RetrievedChunk]], list[float], dict[str, list[RetrievedChunk]] | None]:
        if not self._retrieval_methods:
            return [], [], ({} if collect_per_method else None)

        gathered = await asyncio.gather(
            *(
                method.search(
                    query=query,
                    top_k=method.top_k if method.top_k is not None else fetch_k,
                    filters=filters,
                    knowledge_id=knowledge_id,
                )
                for method in self._retrieval_methods
            )
        )

        result_lists = []
        weights = []
        by_method: dict[str, list[RetrievedChunk]] | None = {} if collect_per_method else None
        for method, results in zip(self._retrieval_methods, gathered, strict=True):
            if by_method is not None:
                by_method[method.name] = list(results) if results else []
            if results:
                result_lists.append(results)
                weights.append(method.weight)
        return result_lists, weights, by_method

    def _apply_source_weights(self, results: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not self._source_type_weights:
            return results
        from dataclasses import replace

        weighted = []
        for r in results:
            weighted.append(replace(r, score=r.score * r.source_weight))
        weighted.sort(key=lambda x: x.score, reverse=True)
        return weighted

    @staticmethod
    def _build_filters(knowledge_id: str | None) -> dict[str, Any] | None:
        if knowledge_id is None:
            return None
        return {"knowledge_id": knowledge_id}
