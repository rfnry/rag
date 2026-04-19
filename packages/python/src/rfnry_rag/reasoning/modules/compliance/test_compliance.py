from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from rfnry_rag.reasoning.common.errors import ComplianceError
from rfnry_rag.reasoning.common.language_model import LanguageModelClient, LanguageModelProvider
from rfnry_rag.reasoning.modules.compliance.models import (
    ComplianceConfig,
    ComplianceDimensionDefinition,
    ComplianceResult,
)
from rfnry_rag.reasoning.modules.compliance.service import ComplianceService


def _lm_client() -> LanguageModelClient:
    return LanguageModelClient(
        provider=LanguageModelProvider(provider="openai", model="gpt-4o-mini", api_key="test-key"),
    )


def _mock_compliant_result() -> SimpleNamespace:
    return SimpleNamespace(
        overall_score=0.95,
        reasoning="fully compliant",
        dimension_scores={"tone": 0.9, "accuracy": 1.0},
        violations=[],
    )


def _mock_noncompliant_result() -> SimpleNamespace:
    return SimpleNamespace(
        overall_score=0.4,
        reasoning="multiple violations found",
        dimension_scores={"authorization": 0.2, "accuracy": 0.6},
        violations=[
            SimpleNamespace(
                dimension="authorization",
                description="exceeded refund limit",
                severity="high",
                suggestion="remove the extra 50%",
            ),
        ],
    )


@patch("rfnry_rag.reasoning.modules.compliance.service.b")
async def test_check_compliant(mock_b):
    """Compliant text returns compliant=True with no violations."""
    mock_b.CheckCompliance = AsyncMock(return_value=_mock_compliant_result())
    service = ComplianceService(lm_client=_lm_client())
    result = await service.check(text="proper response", reference="policy doc")
    assert isinstance(result, ComplianceResult)
    assert result.compliant is True
    assert result.score == pytest.approx(0.95)
    assert result.violations == []


@patch("rfnry_rag.reasoning.modules.compliance.service.b")
async def test_check_noncompliant(mock_b):
    """Non-compliant text returns compliant=False with violations."""
    mock_b.CheckCompliance = AsyncMock(return_value=_mock_noncompliant_result())
    service = ComplianceService(lm_client=_lm_client())
    result = await service.check(
        text="I'll give you 150% refund",
        reference="max refund is 100%",
        config=ComplianceConfig(
            dimensions=[
                ComplianceDimensionDefinition("authorization", "must not exceed limits"),
                ComplianceDimensionDefinition("accuracy", "must be factual"),
            ]
        ),
    )
    assert result.compliant is False
    assert len(result.violations) == 1
    assert result.violations[0].dimension == "authorization"
    assert result.violations[0].severity == "high"
    assert result.violations[0].suggestion == "remove the extra 50%"


@patch("rfnry_rag.reasoning.modules.compliance.service.b")
async def test_check_batch(mock_b):
    """Batch compliance runs concurrently."""
    mock_b.CheckCompliance = AsyncMock(return_value=_mock_compliant_result())
    service = ComplianceService(lm_client=_lm_client())
    results = await service.check_batch(
        [
            ("text 1", "policy 1"),
            ("text 2", "policy 2"),
        ]
    )
    assert len(results) == 2
    assert all(isinstance(r, ComplianceResult) for r in results)


@patch("rfnry_rag.reasoning.modules.compliance.service.b")
async def test_check_truncates_text(mock_b):
    """Text and reference are truncated per config."""
    mock_b.CheckCompliance = AsyncMock(return_value=_mock_compliant_result())
    service = ComplianceService(lm_client=_lm_client())
    await service.check(
        text="x" * 5000,
        reference="y" * 8000,
        config=ComplianceConfig(max_text_length=100, max_reference_length=200),
    )
    call_args = mock_b.CheckCompliance.call_args
    assert len(call_args[0][0]) == 100
    assert len(call_args[0][1]) == 200


@patch("rfnry_rag.reasoning.modules.compliance.service.b")
async def test_check_raises_compliance_error(mock_b):
    """Service wraps unexpected exceptions in ComplianceError."""
    mock_b.CheckCompliance = AsyncMock(side_effect=RuntimeError("boom"))
    service = ComplianceService(lm_client=_lm_client())
    with pytest.raises(ComplianceError, match="Compliance check failed"):
        await service.check(text="test", reference="policy")


@patch("rfnry_rag.reasoning.modules.compliance.service.b")
async def test_threshold_overrides_compliant(mock_b):
    """When threshold is set, compliant is based on score, not violations."""
    mock_b.CheckCompliance = AsyncMock(return_value=_mock_noncompliant_result())
    service = ComplianceService(lm_client=_lm_client())
    result = await service.check(text="test", reference="policy", config=ComplianceConfig(threshold=0.3))
    assert result.compliant is True
    assert len(result.violations) == 1


@patch("rfnry_rag.reasoning.modules.compliance.service.b")
async def test_threshold_fails_below(mock_b):
    """When score is below threshold, compliant is False."""
    mock_b.CheckCompliance = AsyncMock(return_value=_mock_compliant_result())
    service = ComplianceService(lm_client=_lm_client())
    result = await service.check(text="test", reference="policy", config=ComplianceConfig(threshold=0.99))
    assert result.compliant is False


@patch("rfnry_rag.reasoning.modules.compliance.service.b")
async def test_no_threshold_uses_violations(mock_b):
    """Default behavior: compliant only when zero violations."""
    mock_b.CheckCompliance = AsyncMock(return_value=_mock_noncompliant_result())
    service = ComplianceService(lm_client=_lm_client())
    result = await service.check(text="test", reference="policy")
    assert result.compliant is False
    assert len(result.violations) == 1


def test_compliance_config_validation():
    """Config validates required fields."""
    with pytest.raises(ValueError, match="concurrency"):
        ComplianceConfig(concurrency=0)


def test_compliance_config_threshold_validation():
    """Config validates threshold range."""
    with pytest.raises(ValueError, match="threshold"):
        ComplianceConfig(threshold=1.5)
    with pytest.raises(ValueError, match="threshold"):
        ComplianceConfig(threshold=-0.1)
