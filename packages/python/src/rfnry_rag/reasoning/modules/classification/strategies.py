from __future__ import annotations

from typing import Any

from baml_py import ClientRegistry

from rfnry_rag.reasoning.baml.baml_client.async_client import b
from rfnry_rag.reasoning.common.errors import ClassificationError
from rfnry_rag.reasoning.common.logging import get_logger
from rfnry_rag.reasoning.modules.classification.models import CategoryDefinition, Classification
from rfnry_rag.reasoning.protocols import BaseEmbeddings, BaseSemanticIndex

logger = get_logger("classification")


def format_categories(categories: list[CategoryDefinition]) -> str:
    parts = []
    for cat in categories:
        entry = f"- **{cat.name}**: {cat.description}"
        if cat.examples:
            examples = "; ".join(f'"{e}"' for e in cat.examples[:3])
            entry += f"\n  Examples: {examples}"
        parts.append(entry)
    return "\n".join(parts)


async def llm_classify(
    text: str,
    categories: list[CategoryDefinition],
    registry: ClientRegistry,
    max_text_length: int = 2000,
) -> Classification:
    try:
        formatted = format_categories(categories)
        truncated = text[:max_text_length]
        logger.info("[classification/llm] classifying text (%d chars)", len(text))
        result = await b.ClassifyText(
            truncated,
            formatted,
            baml_options={"client_registry": registry},
        )

        category_name = result.category
        valid_names = {c.name for c in categories}
        if category_name not in valid_names:
            lower_map = {c.name.lower(): c.name for c in categories}
            matched = lower_map.get(category_name.lower())
            if matched:
                category_name = matched
            else:
                raise ClassificationError(f"LLM returned invalid category: {category_name}")

        logger.info("[classification/llm] classified as '%s' (confidence: %.2f)", category_name, result.confidence)
        return Classification(
            category=category_name,
            confidence=result.confidence,
            strategy_used="llm",
            reasoning=result.reasoning,
            runner_up=result.runner_up,
            runner_up_confidence=result.runner_up_confidence,
        )
    except ClassificationError:
        raise
    except Exception as exc:
        raise ClassificationError(f"Classification failed: {exc}") from exc


async def knn_classify(
    text: str,
    embeddings: BaseEmbeddings,
    vector_store: BaseSemanticIndex,
    top_k: int,
    knowledge_id: str,
    label_field: str,
) -> Classification:
    try:
        vectors = await embeddings.embed([text])
        return await knn_classify_with_vector(vectors[0], vector_store, top_k, knowledge_id, label_field)
    except ClassificationError:
        raise
    except Exception as exc:
        raise ClassificationError(f"kNN classification failed: {exc}") from exc


async def knn_classify_with_vector(
    vector: list[float],
    vector_store: BaseSemanticIndex,
    top_k: int,
    knowledge_id: str,
    label_field: str,
) -> Classification:
    """Classify using a pre-computed embedding vector (no embed call)."""
    try:
        logger.info("[classification/knn] searching %d neighbors", top_k)
        results = await vector_store.search(
            vector=vector,
            top_k=top_k,
            filters={"knowledge_id": knowledge_id},
        )

        if not results:
            raise ClassificationError("No kNN results found for classification")

        votes: dict[str, int] = {}
        evidence: list[dict[str, Any]] = []
        for r in results:
            payload = r.payload if hasattr(r, "payload") else r
            label = payload.get(label_field, "unknown")
            votes[label] = votes.get(label, 0) + 1
            evidence.append(payload)

        winner = max(votes, key=lambda k: votes[k])
        total = sum(votes.values())
        confidence = votes[winner] / total if total > 0 else 0.0

        runner_up = None
        runner_up_confidence = None
        remaining = {k: v for k, v in votes.items() if k != winner}
        if remaining:
            runner_up = max(remaining, key=lambda k: remaining[k])
            runner_up_confidence = remaining[runner_up] / total

        logger.info("[classification/knn] top category '%s' (%d/%d votes)", winner, votes[winner], total)
        return Classification(
            category=winner,
            confidence=confidence,
            strategy_used="knn",
            runner_up=runner_up,
            runner_up_confidence=runner_up_confidence,
            vote_distribution=votes,
            evidence=evidence,
        )
    except ClassificationError:
        raise
    except Exception as exc:
        raise ClassificationError(f"kNN classification failed: {exc}") from exc


def format_category_sets(sets: list) -> str:
    """Format multiple category sets for the LLM prompt."""
    parts = []
    for s in sets:
        set_header = f"=== Set: {s.name} ==="
        cats = format_categories(s.categories)
        parts.append(f"{set_header}\n{cats}")
    return "\n\n".join(parts)


async def llm_classify_sets(
    text: str,
    sets: list,
    registry: ClientRegistry,
    max_text_length: int = 2000,
) -> dict[str, Classification]:
    """Classify text against multiple category sets in one LLM call. Returns dict keyed by set name."""
    try:
        formatted = format_category_sets(sets)
        truncated = text[:max_text_length]
        logger.info("[classification/llm_sets] classifying text against %d sets", len(sets))
        result = await b.ClassifyTextSets(
            truncated,
            formatted,
            baml_options={"client_registry": registry},
        )

        set_categories = {s.name: {c.name for c in s.categories} for s in sets}
        set_lower_maps = {s.name: {c.name.lower(): c.name for c in s.categories} for s in sets}

        classifications: dict[str, Classification] = {}
        for c in result.classifications:
            category_name = c.category
            valid_names = set_categories.get(c.set_name, set())

            if category_name not in valid_names:
                matched = set_lower_maps.get(c.set_name, {}).get(category_name.lower())
                if matched:
                    category_name = matched
                else:
                    raise ClassificationError(f"LLM returned invalid category '{category_name}' for set '{c.set_name}'")

            classifications[c.set_name] = Classification(
                category=category_name,
                confidence=float(c.confidence),
                strategy_used="llm",
                reasoning=c.reasoning,
                runner_up=c.runner_up,
                runner_up_confidence=float(c.runner_up_confidence) if c.runner_up_confidence else None,
            )

        return classifications
    except ClassificationError:
        raise
    except Exception as exc:
        raise ClassificationError(f"Multi-set classification failed: {exc}") from exc
