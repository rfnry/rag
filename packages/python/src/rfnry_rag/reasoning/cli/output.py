from __future__ import annotations

import json
import sys
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from rfnry_rag.common.cli import OutputMode as OutputMode
from rfnry_rag.common.cli import get_output_mode as get_output_mode

if TYPE_CHECKING:
    from rfnry_rag.reasoning.modules.analysis.models import AnalysisResult
    from rfnry_rag.reasoning.modules.classification.models import Classification
    from rfnry_rag.reasoning.modules.compliance.models import ComplianceResult
    from rfnry_rag.reasoning.modules.evaluation.models import EvaluationResult


def _json_default(obj: Any) -> Any:
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def print_json(data: Any) -> None:
    """Print data as JSON. Handles dataclasses, dicts, and lists."""
    if hasattr(data, "to_dict"):
        data = data.to_dict()
    elif hasattr(data, "__dataclass_fields__"):
        data = asdict(data)
    print(json.dumps(data, default=_json_default, indent=2))


def print_analysis(result: AnalysisResult) -> None:
    """Human-readable analysis result."""
    print(f"Intent: {result.primary_intent} ({result.confidence:.0%} confidence)")

    if result.summary:
        print(f"\nSummary: {result.summary}")

    if result.dimensions:
        print("\nDimensions:")
        for name, dim in result.dimensions.items():
            print(f"  {name}: {dim.value} ({dim.confidence:.0%})")
            if dim.reasoning:
                print(f"    {dim.reasoning}")

    if result.entities:
        print("\nEntities:")
        for e in result.entities:
            print(f'  {e.type}: "{e.value}" — {e.context}')

    if result.retrieval_hints:
        print("\nRetrieval Hints:")
        for h in result.retrieval_hints:
            scope = f" [{h.knowledge_scope}]" if h.knowledge_scope else ""
            print(f"  ({h.priority:.0%}){scope} {h.query}")

    if result.intent_shifts:
        print("\nIntent Shifts:")
        for s in result.intent_shifts:
            print(f"  [{s.at_message}] {s.from_intent} → {s.to_intent}")
            print(f"    {s.reasoning}")

    if result.escalation_detected is not None:
        print(f"\nEscalation: {'yes' if result.escalation_detected else 'no'}")
        if result.escalation_reasoning:
            print(f"  {result.escalation_reasoning}")

    if result.resolution_status:
        print(f"Resolution: {result.resolution_status}")


def print_classification(result: Classification) -> None:
    """Human-readable classification result."""
    print(f"Category: {result.category} ({result.confidence:.0%})")
    print(f"Strategy: {result.strategy_used}")
    if result.reasoning:
        print(f"Reasoning: {result.reasoning}")
    if result.runner_up:
        conf = f" ({result.runner_up_confidence:.0%})" if result.runner_up_confidence is not None else ""
        print(f"Runner-up: {result.runner_up}{conf}")
    if result.needs_review:
        print("*** Needs review (low confidence) ***")


def print_compliance(result: ComplianceResult) -> None:
    """Human-readable compliance result."""
    status = "PASS" if result.compliant else "FAIL"
    print(f"Compliance: {status} ({result.score:.2f})")

    if result.violations:
        print(f"\nViolations ({len(result.violations)}):")
        for v in result.violations:
            print(f"  [{v.severity}] {v.dimension} — {v.description}")
            if v.suggestion:
                print(f"    Suggestion: {v.suggestion}")

    if result.dimension_scores:
        print("\nDimension Scores:")
        for name, score in result.dimension_scores.items():
            print(f"  {name}: {score:.2f}")

    if result.reasoning:
        print(f"\nReasoning: {result.reasoning}")


def print_evaluation(result: EvaluationResult) -> None:
    """Human-readable evaluation result."""
    band = f" ({result.quality_band})" if result.quality_band else ""
    print(f"Score: {result.score:.2f}{band}")

    if result.similarity is not None:
        print(f"Similarity: {result.similarity:.2f}")
    if result.judge_score is not None:
        print(f"Judge: {result.judge_score:.2f}")
    if result.judge_reasoning:
        print(f"  {result.judge_reasoning}")
    if result.dimension_scores:
        print("\nDimension Scores:")
        for name, score in result.dimension_scores.items():
            print(f"  {name}: {score:.2f}")


def print_error(message: str, mode: OutputMode) -> None:
    """Print error in the appropriate format."""
    if mode == OutputMode.JSON:
        print(json.dumps({"error": message}))
    else:
        print(f"Error: {message}", file=sys.stderr)


def print_success(message: str, data: Any, mode: OutputMode) -> None:
    """Print success: JSON data or human message."""
    if mode == OutputMode.JSON:
        print_json(data)
    else:
        print(message)
