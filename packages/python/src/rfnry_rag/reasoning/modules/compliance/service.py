from __future__ import annotations

from rfnry_rag.reasoning.baml.baml_client.async_client import b
from rfnry_rag.reasoning.common.concurrency import run_concurrent
from rfnry_rag.reasoning.common.errors import ComplianceError
from rfnry_rag.reasoning.common.language_model import LanguageModelClient, build_registry
from rfnry_rag.reasoning.common.logging import get_logger
from rfnry_rag.reasoning.modules.compliance.models import (
    ComplianceConfig,
    ComplianceResult,
    Violation,
)

logger = get_logger("compliance")


class ComplianceService:
    """Check text compliance against reference documents."""

    def __init__(self, lm_client: LanguageModelClient) -> None:
        self._registry = build_registry(lm_client)

    async def check(
        self,
        text: str,
        reference: str,
        config: ComplianceConfig | None = None,
    ) -> ComplianceResult:
        """Check a single text for compliance against a reference document."""
        cfg = config or ComplianceConfig()
        try:
            truncated_text = text[: cfg.max_text_length]
            truncated_ref = reference[: cfg.max_reference_length]

            if cfg.dimensions:
                dim_str = ", ".join(f"{d.name} ({d.description})" for d in cfg.dimensions)
            else:
                dim_str = "general compliance"

            logger.info(
                "[compliance/check] checking text (%d chars) against reference (%d chars)",
                len(truncated_text),
                len(truncated_ref),
            )
            result = await b.CheckCompliance(
                truncated_text,
                truncated_ref,
                dim_str,
                baml_options={"client_registry": self._registry},
            )

            violations = [
                Violation(
                    dimension=v.dimension,
                    description=v.description,
                    severity=v.severity,
                    suggestion=v.suggestion,
                )
                for v in (result.violations or [])
            ]

            score = float(result.overall_score)
            compliant = score >= cfg.threshold if cfg.threshold is not None else len(violations) == 0

            logger.info(
                "[compliance/check] score: %.2f, %d violations, compliant: %s",
                score,
                len(violations),
                compliant,
            )
            return ComplianceResult(
                compliant=compliant,
                score=score,
                violations=violations,
                reasoning=result.reasoning,
                dimension_scores={k: float(v) for k, v in result.dimension_scores.items()}
                if result.dimension_scores
                else {},
            )
        except ComplianceError:
            raise
        except Exception as exc:
            raise ComplianceError(f"Compliance check failed: {exc}") from exc

    async def check_batch(
        self,
        items: list[tuple[str, str]],
        config: ComplianceConfig | None = None,
    ) -> list[ComplianceResult]:
        """Check multiple (text, reference) pairs concurrently."""
        cfg = config or ComplianceConfig()

        async def _check_one(item: tuple[str, str]) -> ComplianceResult:
            text, reference = item
            return await self.check(text, reference, cfg)

        return await run_concurrent(items, _check_one, cfg.concurrency)
