from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from rfnry_rag.reasoning.modules.analysis.models import AnalysisConfig, AnalysisResult
from rfnry_rag.reasoning.modules.classification.models import (
    CategoryDefinition,
    Classification,
    ClassificationSetDefinition,
    ClassificationSetResult,
)
from rfnry_rag.reasoning.modules.compliance.models import ComplianceResult, Violation
from rfnry_rag.reasoning.modules.evaluation.models import EvaluationResult
from rfnry_rag.reasoning.modules.pipeline.models import (
    AnalyzeStep,
    ClassifyStep,
    ComplianceStep,
    EvaluateStep,
    PipelineServices,
)
from rfnry_rag.reasoning.modules.pipeline.service import Pipeline


def _analysis_service(intent: str = "inquiry", confidence: float = 0.9) -> MagicMock:
    service = MagicMock()
    service.analyze = AsyncMock(return_value=AnalysisResult(primary_intent=intent, confidence=confidence))
    return service


def _classification_service(category: str = "shipping", confidence: float = 0.85) -> MagicMock:
    service = MagicMock()
    service.classify = AsyncMock(
        return_value=Classification(category=category, confidence=confidence, strategy_used="llm")
    )
    service.classify_sets = AsyncMock(
        return_value=ClassificationSetResult(
            classifications={
                "routing": Classification(category="DELEGATE", confidence=0.9, strategy_used="llm"),
                "topic": Classification(category=category, confidence=confidence, strategy_used="llm"),
            }
        )
    )
    return service


def _evaluation_service(score: float = 0.8) -> MagicMock:
    service = MagicMock()
    service.evaluate = AsyncMock(return_value=EvaluationResult(score=score, similarity=score, quality_band="high"))
    return service


def _compliance_service(compliant: bool = True, score: float = 0.95) -> MagicMock:
    service = MagicMock()
    service.check = AsyncMock(
        return_value=ComplianceResult(compliant=compliant, score=score, violations=[], reasoning="ok")
    )
    return service


async def test_full_intake_pipeline():
    pipeline = Pipeline(
        services=PipelineServices(
            analysis=_analysis_service(),
            classification=_classification_service(),
        )
    )
    categories = [CategoryDefinition("shipping", "delivery"), CategoryDefinition("billing", "payment")]
    result = await pipeline.run(
        "where is my order",
        steps=[
            AnalyzeStep(config=AnalysisConfig(summarize=True)),
            ClassifyStep(categories=categories),
        ],
    )
    assert result.analysis.primary_intent == "inquiry"
    assert result.classification.category == "shipping"
    assert result.duration_ms > 0


async def test_full_intake_with_sets():
    pipeline = Pipeline(
        services=PipelineServices(
            analysis=_analysis_service(),
            classification=_classification_service(),
        )
    )
    sets = [
        ClassificationSetDefinition("routing", [CategoryDefinition("DELEGATE", "route to agent")]),
        ClassificationSetDefinition("topic", [CategoryDefinition("shipping", "delivery")]),
    ]
    result = await pipeline.run(
        "where is my order",
        steps=[AnalyzeStep(), ClassifyStep(sets=sets)],
    )
    assert isinstance(result.classification, ClassificationSetResult)
    assert "routing" in result.classification.classifications
    assert "topic" in result.classification.classifications


async def test_evaluate_then_compliance():
    pipeline = Pipeline(
        services=PipelineServices(
            evaluation=_evaluation_service(),
            compliance=_compliance_service(),
        )
    )
    result = await pipeline.run(
        "the generated response",
        steps=[
            EvaluateStep(reference="the reference answer"),
            ComplianceStep(reference="company policy document"),
        ],
    )
    assert result.evaluation.score == pytest.approx(0.8)
    assert result.compliance.compliant is True


async def test_pipeline_compliance_failure():
    service = MagicMock()
    service.check = AsyncMock(
        return_value=ComplianceResult(
            compliant=False,
            score=0.3,
            violations=[
                Violation(dimension="tone", description="too casual", severity="high", suggestion="be professional")
            ],
            reasoning="failed",
        )
    )
    pipeline = Pipeline(services=PipelineServices(compliance=service))
    result = await pipeline.run("casual reply lol", steps=[ComplianceStep(reference="be professional")])
    assert result.compliance.compliant is False
    assert len(result.compliance.violations) == 1
    assert result.compliance.violations[0].severity == "high"


async def test_pipeline_all_steps():
    pipeline = Pipeline(
        services=PipelineServices(
            analysis=_analysis_service(),
            classification=_classification_service(),
            evaluation=_evaluation_service(),
            compliance=_compliance_service(),
        )
    )
    categories = [CategoryDefinition("shipping", "delivery")]
    result = await pipeline.run(
        "test input",
        steps=[
            AnalyzeStep(),
            ClassifyStep(categories=categories),
            EvaluateStep(reference="ref"),
            ComplianceStep(reference="policy"),
        ],
    )
    assert result.analysis is not None
    assert result.classification is not None
    assert result.evaluation is not None
    assert result.compliance is not None
    assert result.duration_ms > 0


async def test_missing_classification_service():
    pipeline = Pipeline(services=PipelineServices(analysis=_analysis_service()))
    categories = [CategoryDefinition("a", "desc")]
    with pytest.raises(ValueError, match="requires classification service"):
        await pipeline.run("test", steps=[ClassifyStep(categories=categories)])


async def test_missing_evaluation_service():
    pipeline = Pipeline(services=PipelineServices())
    with pytest.raises(ValueError, match="requires evaluation service"):
        await pipeline.run("test", steps=[EvaluateStep(reference="ref")])


async def test_missing_compliance_service():
    pipeline = Pipeline(services=PipelineServices())
    with pytest.raises(ValueError, match="requires compliance service"):
        await pipeline.run("test", steps=[ComplianceStep(reference="policy")])


def test_classify_step_requires_categories_or_sets():
    with pytest.raises(ValueError, match="requires either categories or sets"):
        ClassifyStep()


def test_classify_step_accepts_categories():
    step = ClassifyStep(categories=[CategoryDefinition("a", "desc")])
    assert step.categories is not None


def test_classify_step_accepts_sets():
    step = ClassifyStep(sets=[ClassificationSetDefinition("s", [CategoryDefinition("a", "desc")])])
    assert step.sets is not None
