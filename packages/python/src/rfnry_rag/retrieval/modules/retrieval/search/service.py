import asyncio
from typing import Any

from rfnry_rag.retrieval.common.logging import get_logger
from rfnry_rag.retrieval.common.models import RetrievedChunk
from rfnry_rag.retrieval.modules.retrieval.base import BaseRetrievalMethod
from rfnry_rag.retrieval.modules.retrieval.refinement.base import BaseChunkRefinement
from rfnry_rag.retrieval.modules.retrieval.search.fusion import reciprocal_rank_fusion
from rfnry_rag.retrieval.modules.retrieval.search.reranking.base import BaseReranking
from rfnry_rag.retrieval.modules.retrieval.search.rewriting.base import BaseQueryRewriting

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

    async def retrieve(
        self,
        query: str,
        knowledge_id: str | None = None,
        top_k: int | None = None,
        tree_chunks: list[RetrievedChunk] | None = None,
    ) -> list[RetrievedChunk]:
        if not query or not query.strip():
            return []

        top_k = top_k if top_k is not None else self._top_k
        fetch_k = top_k * 4
        filters = self._build_filters(knowledge_id)

        logger.info('query: "%s" (knowledge_id=%s)', query[:80], knowledge_id)

        queries = [query]
        if self._query_rewriter:
            try:
                rewritten = await self._query_rewriter.rewrite(query)
                queries.extend(rewritten)
                if rewritten:
                    logger.info(
                        "query rewriting: %d total queries (1 original + %d rewritten)",
                        len(queries),
                        len(rewritten),
                    )
            except Exception as exc:
                logger.exception("query rewriter failed: %s — proceeding with original query", exc)

        search_tasks = [self._search_single_query(q, fetch_k, filters, knowledge_id) for q in queries]
        query_results = await asyncio.gather(*search_tasks, return_exceptions=True)

        all_result_lists: list[list[RetrievedChunk]] = []
        all_weights: list[float] = []
        for idx, outcome in enumerate(query_results):
            if isinstance(outcome, BaseException):
                logger.warning("query variant %d failed: %s — skipping", idx, outcome)
                continue
            result_lists, weights = outcome
            all_result_lists.extend(result_lists)
            all_weights.extend(weights)

        if tree_chunks:
            all_result_lists.append(tree_chunks)
            all_weights.append(1.0)
            logger.info("%d tree search candidates added to fusion", len(tree_chunks))

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

        if self._reranking and fused:
            fused = await self._reranking.rerank(query, fused, top_k=top_k)
            logger.info("top %d selected after reranking", len(fused))
        else:
            fused = fused[:top_k]

        if self._chunk_refiner and fused:
            fused = await self._chunk_refiner.refine(query, fused)
            logger.info("chunk refinement: %d chunks after refinement", len(fused))

        return fused

    async def _search_single_query(
        self,
        query: str,
        fetch_k: int,
        filters: dict[str, Any] | None,
        knowledge_id: str | None,
    ) -> tuple[list[list[RetrievedChunk]], list[float]]:
        """Run all retrieval methods in parallel for a single query."""
        if not self._retrieval_methods:
            return [], []

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
        for method, results in zip(self._retrieval_methods, gathered, strict=True):
            if results:
                result_lists.append(results)
                weights.append(method.weight)
        return result_lists, weights

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
