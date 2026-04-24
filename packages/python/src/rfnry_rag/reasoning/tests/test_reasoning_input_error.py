"""Regression tests for ReasoningInputError: inheritance chain and back-compat."""

from __future__ import annotations

import pytest


def test_reasoning_input_error_catchable_as_both_parents() -> None:
    """ReasoningInputError must satisfy isinstance checks for both parent classes."""
    from rfnry_rag.reasoning.common.errors import ReasoningError, ReasoningInputError

    err = ReasoningInputError("x")
    assert isinstance(err, ReasoningError), "must be catchable as ReasoningError"
    assert isinstance(err, ValueError), "must be catchable as ValueError (back-compat)"


def test_reasoning_input_error_str() -> None:
    """The error message must be preserved."""
    from rfnry_rag.reasoning.common.errors import ReasoningInputError

    err = ReasoningInputError("bad config value")
    assert "bad config value" in str(err)


def test_classification_config_raises_typed_input_error_not_bare_valueerror() -> None:
    """ClassificationConfig must raise ReasoningInputError on invalid max_text_length."""
    from rfnry_rag.reasoning.common.errors import ReasoningInputError
    from rfnry_rag.reasoning.modules.classification.models import ClassificationConfig

    with pytest.raises(ReasoningInputError, match="max_text_length"):
        ClassificationConfig(max_text_length=10_000_000)  # exceeds 5M ceiling

    # Back-compat: existing `except ValueError:` still catches ReasoningInputError
    with pytest.raises(ValueError):
        ClassificationConfig(max_text_length=10_000_000)


def test_analysis_config_raises_typed_input_error() -> None:
    """AnalysisConfig must raise ReasoningInputError on invalid concurrency."""
    from rfnry_rag.reasoning.common.errors import ReasoningInputError
    from rfnry_rag.reasoning.modules.analysis.models import AnalysisConfig

    with pytest.raises(ReasoningInputError, match="concurrency"):
        AnalysisConfig(concurrency=0)

    with pytest.raises(ValueError):
        AnalysisConfig(concurrency=0)


def test_clustering_config_raises_typed_input_error() -> None:
    """ClusteringConfig must raise ReasoningInputError on invalid n_clusters."""
    from rfnry_rag.reasoning.common.errors import ReasoningInputError
    from rfnry_rag.reasoning.modules.clustering.models import ClusteringConfig

    with pytest.raises(ReasoningInputError, match="n_clusters"):
        ClusteringConfig(algorithm="kmeans", n_clusters=1)

    with pytest.raises(ValueError):
        ClusteringConfig(algorithm="kmeans", n_clusters=1)


def test_compliance_config_raises_typed_input_error() -> None:
    """ComplianceConfig must raise ReasoningInputError on invalid concurrency."""
    from rfnry_rag.reasoning.common.errors import ReasoningInputError
    from rfnry_rag.reasoning.modules.compliance.models import ComplianceConfig

    with pytest.raises(ReasoningInputError, match="concurrency"):
        ComplianceConfig(concurrency=0)

    with pytest.raises(ValueError):
        ComplianceConfig(concurrency=0)


def test_evaluation_config_raises_typed_input_error() -> None:
    """EvaluationConfig must raise ReasoningInputError on invalid strategy."""
    from rfnry_rag.reasoning.common.errors import ReasoningInputError
    from rfnry_rag.reasoning.modules.evaluation.models import EvaluationConfig

    with pytest.raises(ReasoningInputError, match="Unknown strategy"):
        EvaluationConfig(strategy="bad")  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        EvaluationConfig(strategy="bad")  # type: ignore[arg-type]


def test_reasoning_input_error_exported_from_reasoning_init() -> None:
    """ReasoningInputError must be importable from the top-level reasoning package."""
    from rfnry_rag.reasoning import ReasoningInputError  # noqa: F401


def test_reasoning_input_error_exported_from_top_level() -> None:
    """ReasoningInputError must be importable from the top-level rfnry_rag package."""
    from rfnry_rag import ReasoningInputError  # noqa: F401
