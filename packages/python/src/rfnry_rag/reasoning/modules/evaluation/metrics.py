from __future__ import annotations

import numpy as np
from baml_py import ClientRegistry

from rfnry_rag.reasoning.baml.baml_client.async_client import b
from rfnry_rag.reasoning.common.errors import EvaluationError
from rfnry_rag.reasoning.common.logging import get_logger
from rfnry_rag.reasoning.protocols import BaseEmbeddings

logger = get_logger("evaluation")


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    a = np.asarray(vec_a, dtype=np.float32)
    b_ = np.asarray(vec_b, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b_)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b_) / (norm_a * norm_b))


async def semantic_similarity(
    text_a: str,
    text_b: str,
    embeddings: BaseEmbeddings,
) -> float:
    logger.info("[evaluation/similarity] comparing texts (%d, %d chars)", len(text_a), len(text_b))
    vectors = await embeddings.embed([text_a, text_b])
    score = cosine_similarity(vectors[0], vectors[1])
    logger.info("[evaluation/similarity] score: %.4f", score)
    return score


async def llm_judge(
    generated: str,
    reference: str,
    registry: ClientRegistry,
    dimensions: list[str],
    context: str | None = None,
    max_text_length: int = 3000,
) -> tuple[float, str | None, dict[str, float] | None]:
    try:
        logger.info("[evaluation/judge] judging output (%d chars)", len(generated))
        result = await b.JudgeOutput(
            generated[:max_text_length],
            reference[:max_text_length],
            ", ".join(dimensions) if dimensions else "overall quality",
            context,
            baml_options={"client_registry": registry},
        )
        logger.info("[evaluation/judge] score: %.2f", result.overall_score)
        return (
            float(result.overall_score),
            result.reasoning,
            {k: float(v) for k, v in result.dimension_scores.items()} if result.dimension_scores else None,
        )
    except Exception as exc:
        raise EvaluationError(f"LLM judge failed: {exc}") from exc
