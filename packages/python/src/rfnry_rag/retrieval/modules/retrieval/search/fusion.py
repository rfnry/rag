from dataclasses import replace

from rfnry_rag.retrieval.common.models import RetrievedChunk


def reciprocal_rank_fusion(
    result_lists: list[list[RetrievedChunk]],
    k: int = 60,
    source_type_weights: dict[str, float] | None = None,
    method_weights: list[float] | None = None,
) -> list[RetrievedChunk]:
    scores: dict[str, float] = {}
    results_by_key: dict[str, RetrievedChunk] = {}

    for list_idx, result_list in enumerate(result_lists):
        list_weight = method_weights[list_idx] if method_weights and list_idx < len(method_weights) else 1.0
        for rank, result in enumerate(result_list):
            key = result.chunk_id
            if key not in scores:
                scores[key] = 0
                results_by_key[key] = result
            scores[key] += list_weight / (k + rank + 1)

    if source_type_weights:
        for key, result in results_by_key.items():
            scores[key] *= result.source_weight

    sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    fused = []
    for key in sorted_keys:
        fused.append(replace(results_by_key[key], score=scores[key]))

    return fused
