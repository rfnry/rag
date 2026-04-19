from __future__ import annotations

from baml_py import ClientRegistry

from rfnry_rag.reasoning.baml.baml_client.async_client import b
from rfnry_rag.reasoning.common.concurrency import run_concurrent
from rfnry_rag.reasoning.common.logging import get_logger

logger = get_logger("clustering")


async def generate_cluster_labels(
    sample_groups: list[list[str]],
    registry: ClientRegistry,
    max_concurrent: int = 10,
) -> list[str | None]:
    async def _label_one(samples: list[str]) -> str | None:
        if not samples:
            return None
        formatted = "\n".join(f"- {s[:500]}" for s in samples[:10])
        try:
            logger.info("[clustering/label] generating label for %d samples", len(samples))
            result = await b.LabelCluster(
                formatted,
                baml_options={"client_registry": registry},
            )
            logger.info("[clustering/label] generated: '%s'", result.label)
            return result.label
        except Exception:
            logger.exception("[clustering/label] failed to generate label")
            return None

    return await run_concurrent(sample_groups, _label_one, max_concurrent)
