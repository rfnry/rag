from dataclasses import replace

from baml_py import errors as baml_errors

from rfnry_rag.common.language_model import LanguageModelClient, build_registry
from rfnry_rag.common.logging import get_logger
from rfnry_rag.retrieval.common.models import RetrievedChunk

logger = get_logger(__name__)


class _LLMReranking:
    def __init__(self, lm_client: LanguageModelClient) -> None:
        self._registry = build_registry(lm_client)

    async def rerank(self, query: str, results: list[RetrievedChunk], top_k: int = 5) -> list[RetrievedChunk]:
        if not results:
            return []

        from rfnry_rag.baml.baml_client.async_client import b

        passages = "\n".join(f"[{i}] {r.content}" for i, r in enumerate(results))

        try:
            scores_result = await b.RerankChunks(
                query,
                passages,
                baml_options={"client_registry": self._registry},
            )

            scores = [s.score for s in scores_result]

            if len(scores) != len(results):
                logger.warning(
                    "BAML reranker returned %d scores for %d results, falling back",
                    len(scores),
                    len(results),
                )
                return results[:top_k]

        except baml_errors.BamlValidationError as exc:
            logger.exception(
                "RerankChunks failed: LLM returned an unparseable response — returning unranked results. Detail: %s",
                exc,
            )
            return results[:top_k]
        except Exception as exc:
            logger.exception(
                "RerankChunks failed: %s — returning unranked results",
                exc,
            )
            return results[:top_k]

        raw_max = max(scores) if scores else 1
        max_score = raw_max if raw_max > 0 else 1

        scored = []
        for result, score in zip(results, scores, strict=True):
            scored.append(replace(result, score=max(0.0, float(score)) / max_score))

        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]
