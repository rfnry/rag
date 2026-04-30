from __future__ import annotations

import asyncio
import time
from typing import Any

from rfnry_rag.common.logging import get_logger, query_logging_enabled
from rfnry_rag.retrieval.base import BaseRetrievalMethod
from rfnry_rag.retrieval.common.models import RetrievalTrace, RetrievedChunk
from rfnry_rag.retrieval.refinement.base import BaseChunkRefinement
from rfnry_rag.retrieval.search.fusion import reciprocal_rank_fusion
from rfnry_rag.retrieval.search.reranking.base import BaseReranking
from rfnry_rag.retrieval.search.rewriting.base import BaseQueryRewriting

logger = get_logger("retrieval.search.service")


class RetrievalService:
    def __init__(
        self,
        retrieval_methods: list[BaseRetrievalMethod],
        reranking: BaseReranking | None = None,
        top_k: int = 5,
        source_type_weights: dict[str, float] | None = None,
        query_rewriter: BaseQueryRewriting | None = None,
        chunk_refiner: BaseChunkRefinement | None = None,
    ) -> None:
        self._retrieval_methods = retrieval_methods
        self._reranking = reranking
        self._top_k = top_k
        self._source_type_weights = source_type_weights
        self._query_rewriter = query_rewriter
        self._chunk_refiner = chunk_refiner

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

        queries = [query]
        if self._query_rewriter:
            rewriting_start = time.perf_counter() if trace_obj is not None else 0.0
            try:
                rewritten = await self._query_rewriter.rewrite(query)
                queries.extend(rewritten)
                if rewritten:
                    logger.info(
                        "query rewriting: %d total queries (1 original + %d rewritten)",
                        len(queries),
                        len(rewritten),
                    )
                if trace_obj is not None:
                    trace_obj.rewritten_queries = list(rewritten)
            except Exception as exc:
                logger.exception("query rewriter failed: %s — proceeding with original query", exc)
                # Rewriter failure leaves rewritten_queries=[] (not the partial
                # variants) — the trace records "rewriter ran and produced no
                # usable variants".
            if trace_obj is not None:
                trace_obj.timings["rewriting"] = time.perf_counter() - rewriting_start

        retrieval_start = time.perf_counter() if trace_obj is not None else 0.0
        search_tasks = [
            self._search_single_query(
                q,
                fetch_k,
                filters,
                knowledge_id,
                collect_per_method=trace_obj is not None,
            )
            for q in queries
        ]
        query_results = await asyncio.gather(*search_tasks, return_exceptions=True)

        all_result_lists: list[list[RetrievedChunk]] = []
        all_weights: list[float] = []
        successes = 0
        # Per-method aggregation seeded with every declared method so the
        # "ran-and-empty" vs "not configured" distinction survives even when
        # every variant returned [] for a given method.
        per_method: dict[str, list[RetrievedChunk]] = (
            {m.name: [] for m in self._retrieval_methods} if trace_obj is not None else {}
        )
        for idx, outcome in enumerate(query_results):
            if isinstance(outcome, BaseException):
                logger.warning("query variant %d failed: %s — skipping", idx, outcome)
                continue
            successes += 1
            result_lists, weights, by_method = outcome
            all_result_lists.extend(result_lists)
            all_weights.extend(weights)
            if trace_obj is not None and by_method is not None:
                for method_name, results in by_method.items():
                    per_method.setdefault(method_name, []).extend(results)

        if query_results and successes == 0:
            from rfnry_rag.exceptions import RetrievalError

            raise RetrievalError("all retrieval query variants failed")

        if trace_obj is not None:
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

        if self._chunk_refiner:
            refinement_start = time.perf_counter() if trace_obj is not None else 0.0
            if fused:
                fused = await self._chunk_refiner.refine(query, fused)
                logger.info("chunk refinement: %d chunks after refinement", len(fused))
            if trace_obj is not None:
                trace_obj.refined_results = list(fused)
                trace_obj.timings["refinement"] = time.perf_counter() - refinement_start

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
