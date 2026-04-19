from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from rfnry_rag.reasoning.modules.analysis.models import AnalysisResult
from rfnry_rag.reasoning.modules.classification.models import (
    CategoryDefinition,
    Classification,
    ClassificationSetDefinition,
    ClassificationSetResult,
)
from rfnry_rag.reasoning.modules.compliance.models import ComplianceResult
from rfnry_rag.reasoning.modules.evaluation.models import EvaluationResult
from rfnry_rag.reasoning.modules.pipeline.models import (
    AnalyzeStep,
    ClassifyStep,
    ComplianceStep,
    EvaluateStep,
    PipelineResult,
    PipelineServices,
)
from rfnry_rag.reasoning.modules.pipeline.service import Pipeline


def _mock_analysis_service() -> MagicMock:
    service = MagicMock()
    service.analyze = AsyncMock(return_value=AnalysisResult(primary_intent="inquiry", confidence=0.9))
    return service


def _mock_classification_service() -> MagicMock:
    service = MagicMock()
    service.classify = AsyncMock(return_value=Classification(category="shipping", confidence=0.85, strategy_used="llm"))
    service.classify_sets = AsyncMock(
        return_value=ClassificationSetResult(
            classifications={"topic": Classification(category="shipping", confidence=0.85, strategy_used="llm")}
        )
    )
    return service


def _mock_evaluation_service() -> MagicMock:
    service = MagicMock()
    service.evaluate = AsyncMock(return_value=EvaluationResult(score=0.8, similarity=0.8, quality_band="high"))
    return service


def _mock_compliance_service() -> MagicMock:
    service = MagicMock()
    service.check = AsyncMock(
        return_value=ComplianceResult(compliant=True, score=0.95, violations=[], reasoning="ok", dimension_scores={})
    )
    return service


async def test_pipeline_analyze_step():
    """Pipeline runs analysis step and stores result."""
    pipeline = Pipeline(services=PipelineServices(analysis=_mock_analysis_service()))
    result = await pipeline.run("test text", steps=[AnalyzeStep()])
    assert isinstance(result, PipelineResult)
    assert result.analysis is not None
    assert result.analysis.primary_intent == "inquiry"


async def test_pipeline_classify_step_single():
    """Pipeline runs single classification step."""
    categories = [CategoryDefinition("a", "desc a"), CategoryDefinition("b", "desc b")]
    pipeline = Pipeline(services=PipelineServices(classification=_mock_classification_service()))
    result = await pipeline.run("test", steps=[ClassifyStep(categories=categories)])
    assert result.classification is not None
    assert isinstance(result.classification, Classification)


async def test_pipeline_classify_step_sets():
    """Pipeline runs multi-set classification step."""
    sets = [ClassificationSetDefinition("topic", [CategoryDefinition("a", "desc")])]
    pipeline = Pipeline(services=PipelineServices(classification=_mock_classification_service()))
    result = await pipeline.run("test", steps=[ClassifyStep(sets=sets)])
    assert result.classification is not None
    assert isinstance(result.classification, ClassificationSetResult)


async def test_pipeline_evaluate_step():
    """Pipeline runs evaluation step."""
    pipeline = Pipeline(services=PipelineServices(evaluation=_mock_evaluation_service()))
    result = await pipeline.run("generated text", steps=[EvaluateStep(reference="reference text")])
    assert result.evaluation is not None
    assert result.evaluation.score == pytest.approx(0.8)


async def test_pipeline_compliance_step():
    """Pipeline runs compliance step."""
    pipeline = Pipeline(services=PipelineServices(compliance=_mock_compliance_service()))
    result = await pipeline.run("response text", steps=[ComplianceStep(reference="policy doc")])
    assert result.compliance is not None
    assert result.compliance.compliant is True


async def test_pipeline_multiple_steps():
    """Pipeline runs multiple steps sequentially."""
    pipeline = Pipeline(
        services=PipelineServices(
            analysis=_mock_analysis_service(),
            classification=_mock_classification_service(),
        )
    )
    categories = [CategoryDefinition("a", "desc")]
    result = await pipeline.run(
        "test text",
        steps=[AnalyzeStep(), ClassifyStep(categories=categories)],
    )
    assert result.analysis is not None
    assert result.classification is not None
    assert result.duration_ms > 0


async def test_pipeline_missing_service():
    """Pipeline raises when step requires a service not provided."""
    pipeline = Pipeline(services=PipelineServices())
    with pytest.raises(ValueError, match="requires analysis service"):
        await pipeline.run("test", steps=[AnalyzeStep()])


def test_classify_step_requires_categories_or_sets():
    """ClassifyStep must have categories or sets."""
    with pytest.raises(ValueError, match="requires either categories or sets"):
        ClassifyStep()
