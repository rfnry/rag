from __future__ import annotations

import time

from rfnry_rag.reasoning.common.logging import get_logger
from rfnry_rag.reasoning.modules.evaluation.models import EvaluationPair
from rfnry_rag.reasoning.modules.pipeline.models import (
    AnalyzeStep,
    ClassifyStep,
    ComplianceStep,
    EvaluateStep,
    PipelineResult,
    PipelineServices,
    PipelineStep,
)

logger = get_logger("pipeline")


class Pipeline:
    """Compose SDK modules into sequential flows.

    Supported step types: AnalyzeStep, ClassifyStep, EvaluateStep, ComplianceStep.

    **Clustering is intentionally excluded.** ClusteringService operates over
    a corpus (list[str]) and produces labeled groups, which does not fit
    the single-text step chain that Pipeline models. Use `ClusteringService`
    directly for clustering flows; compose its output into a Pipeline by
    selecting representative texts per cluster and feeding them through.
    """

    def __init__(self, services: PipelineServices) -> None:
        self._services = services

    async def run(self, text: str, steps: list[PipelineStep]) -> PipelineResult:
        """Run steps sequentially. Returns accumulated results from all steps."""
        start = time.monotonic()
        result = PipelineResult()

        for step in steps:
            if isinstance(step, AnalyzeStep):
                if not self._services.analysis:
                    raise ValueError("AnalyzeStep requires analysis service in PipelineServices")
                logger.info("[pipeline] running analyze step")
                result.analysis = await self._services.analysis.analyze(text, step.config)

            elif isinstance(step, ClassifyStep):
                if not self._services.classification:
                    raise ValueError("ClassifyStep requires classification service in PipelineServices")
                logger.info("[pipeline] running classify step")
                if step.sets:
                    result.classification = await self._services.classification.classify_sets(
                        text, step.sets, step.config
                    )
                elif step.categories:
                    result.classification = await self._services.classification.classify(
                        text, step.categories, step.config
                    )

            elif isinstance(step, EvaluateStep):
                if not self._services.evaluation:
                    raise ValueError("EvaluateStep requires evaluation service in PipelineServices")
                logger.info("[pipeline] running evaluate step")
                result.evaluation = await self._services.evaluation.evaluate(
                    EvaluationPair(generated=text, reference=step.reference),
                    step.config,
                )

            elif isinstance(step, ComplianceStep):
                if not self._services.compliance:
                    raise ValueError("ComplianceStep requires compliance service in PipelineServices")
                logger.info("[pipeline] running compliance step")
                result.compliance = await self._services.compliance.check(text, step.reference, step.config)

        result.duration_ms = (time.monotonic() - start) * 1000
        logger.info("[pipeline] completed %d steps in %.1fms", len(steps), result.duration_ms)
        return result
