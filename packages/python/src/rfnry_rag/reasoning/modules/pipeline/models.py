from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rfnry_rag.reasoning.modules.analysis.models import AnalysisConfig, AnalysisResult
    from rfnry_rag.reasoning.modules.analysis.service import AnalysisService
    from rfnry_rag.reasoning.modules.classification.models import (
        CategoryDefinition,
        Classification,
        ClassificationConfig,
        ClassificationSetDefinition,
        ClassificationSetResult,
    )
    from rfnry_rag.reasoning.modules.classification.service import ClassificationService
    from rfnry_rag.reasoning.modules.compliance.models import ComplianceConfig, ComplianceResult
    from rfnry_rag.reasoning.modules.compliance.service import ComplianceService
    from rfnry_rag.reasoning.modules.evaluation.models import EvaluationConfig, EvaluationResult
    from rfnry_rag.reasoning.modules.evaluation.service import EvaluationService


@dataclass
class PipelineServices:
    """Services available to the pipeline. Only provide what you need."""

    analysis: AnalysisService | None = None
    classification: ClassificationService | None = None
    evaluation: EvaluationService | None = None
    compliance: ComplianceService | None = None


@dataclass
class AnalyzeStep:
    """Pipeline step: analyze input text."""

    config: AnalysisConfig | None = None


@dataclass
class ClassifyStep:
    """Pipeline step: classify input text."""

    categories: list[CategoryDefinition] | None = None
    sets: list[ClassificationSetDefinition] | None = None
    config: ClassificationConfig | None = None

    def __post_init__(self) -> None:
        if not self.categories and not self.sets:
            raise ValueError("ClassifyStep requires either categories or sets")


@dataclass
class EvaluateStep:
    """Pipeline step: evaluate input text against a reference."""

    reference: str
    config: EvaluationConfig | None = None


@dataclass
class ComplianceStep:
    """Pipeline step: check input text compliance against a reference."""

    reference: str
    config: ComplianceConfig | None = None


PipelineStep = AnalyzeStep | ClassifyStep | EvaluateStep | ComplianceStep


@dataclass
class PipelineResult:
    """Accumulated results from all pipeline steps."""

    analysis: AnalysisResult | None = None
    classification: Classification | ClassificationSetResult | None = None
    evaluation: EvaluationResult | None = None
    compliance: ComplianceResult | None = None
    duration_ms: float = 0.0
