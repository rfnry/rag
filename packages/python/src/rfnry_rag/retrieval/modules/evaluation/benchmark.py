"""Benchmark harness for retrieval + generation quality (R8.3).

Pure orchestration over R8.1's `RetrievalTrace` and R8.2's
`classify_failure`. Reuses the existing `ExactMatch` and `F1Score` from
`metrics.py`, and (when configured) `LLMJudgment` for per-case judging.
Adds a source-id-based `retrieval_recall` / `retrieval_precision`
computation distinct from the content-based `RetrievalRecall` /
`RetrievalPrecision` in `retrieval_metrics.py` — this measures whether
the benchmark's `expected_source_ids` were retrieved, not chunk-content
overlap with the expected answer. No new metric implementations and no
new LLM calls beyond what the configured metrics already do.

R8.3 ships the user-facing payoff of R8.1 + R8.2: aggregate metrics
(EM / F1 / optional LLM-judge / retrieval recall+precision) PLUS
per-case traces PLUS failure-class distribution, in one report.

Failure rule
------------
A case is judged a failure when *either*:

1. F1 is strictly below `BenchmarkConfig.failure_threshold` (default 0.5
   — an honest mid-point: anything weaker is too generous, anything
   stricter mis-classifies legitimate paraphrases as failures), OR
2. The trace's `grounding_decision == "ungrounded"`. This second check
   catches the small-but-real "F1 was technically high but grounding
   still flagged it ungrounded" category — e.g. the LLM produced a
   plausible-sounding answer that the grounding gate already rejected.

Failures (and only failures) are passed through `classify_failure` from
R8.2, and the resulting `FailureType` is aggregated into a
`failure_distribution` histogram.

`failure_distribution` keying
-----------------------------
The histogram keys are `FailureType.name` strings (e.g.
`"VOCABULARY_MISMATCH"`), not the enum or the `FailureClassification`
dataclass. `FailureClassification` is `frozen=True` but contains a
`dict[str, ...]` field, so Python does not generate `__hash__` for it
and using it as a `Counter` key would raise `TypeError: unhashable type`
at runtime. Keying on the name (rather than the enum directly) also
makes the JSON output of `--output report.json` human-readable without
post-processing.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rfnry_rag.common.concurrency import run_concurrent
from rfnry_rag.common.errors import ConfigurationError
from rfnry_rag.retrieval.common.logging import get_logger
from rfnry_rag.retrieval.modules.evaluation.failure_analysis import (
    FailureClassification,
    classify_failure,
)
from rfnry_rag.retrieval.modules.evaluation.metrics import ExactMatch, F1Score, LLMJudgment

if TYPE_CHECKING:
    from rfnry_rag.retrieval.modules.generation.models import QueryResult

logger = get_logger("evaluation.benchmark")

# Caller passes an `async def fn(text: str, *, trace: bool) -> QueryResult`.
# Typed as `Callable[..., Awaitable[QueryResult]]` (rather than a
# Protocol) so a bound method or a free async function both satisfy it
# without ceremony.
_QueryFn = Callable[..., Awaitable["QueryResult"]]


@dataclass
class BenchmarkCase:
    """One evaluation case: query + expected answer (+ optional expected source IDs).

    `expected_source_ids` is optional. When supplied for every case, the
    report exposes retrieval recall+precision; when `None` for any case,
    those aggregates report `None` rather than 0.0 (the alternative
    would conflate "not applicable" with "zero hits").
    """

    query: str
    expected_answer: str
    expected_source_ids: list[str] | None = None


@dataclass
class BenchmarkConfig:
    """Knobs for `RagEngine.benchmark`. Both fields are bounded.

    `concurrency` defaults to 1 (serial) so a small benchmark behaves
    identically to a `for case in cases: ...` loop. Larger benchmarks
    opt into parallelism via `run_concurrent` (the same helper R3 uses).
    """

    concurrency: int = 1
    failure_threshold: float = 0.5

    def __post_init__(self) -> None:
        if not (1 <= self.concurrency <= 20):
            raise ConfigurationError(
                f"BenchmarkConfig.concurrency={self.concurrency} out of range [1, 20]"
            )
        if not (0.0 <= self.failure_threshold <= 1.0):
            raise ConfigurationError(
                f"BenchmarkConfig.failure_threshold={self.failure_threshold} out of range [0.0, 1.0]"
            )


@dataclass
class BenchmarkCaseResult:
    """Per-case outcome carrying the full `QueryResult` (with `trace`) plus metrics.

    `failure` is `None` when the case passed; populated via
    `classify_failure` (R8.2) when the case is judged a failure. The
    raw `QueryResult` is retained so consumers writing benchmarks can
    drill into the trace for any case (the CLI's `--output` JSON
    flag round-trips this entire structure for offline inspection).
    """

    case: BenchmarkCase
    result: QueryResult
    failure: FailureClassification | None
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class BenchmarkReport:
    """Aggregate report for a benchmark run.

    `retrieval_recall` and `retrieval_precision` are `None` when at
    least one case lacks `expected_source_ids` — N/A is distinct from
    0.0. `llm_judge_score` is `None` when the engine has no LLM judge
    configured (judging is opt-in; we do not silently spend tokens).
    """

    total_cases: int
    retrieval_recall: float | None
    retrieval_precision: float | None
    generation_em: float
    generation_f1: float
    llm_judge_score: float | None
    failure_distribution: dict[str, int]
    per_case_results: list[BenchmarkCaseResult]


def _is_failure(case: BenchmarkCase, result: QueryResult, threshold: float) -> bool:
    """Documented failure rule: low F1 OR explicitly ungrounded.

    See module docstring for the threshold-rationale and why the
    grounding-ungrounded path is a distinct failure trigger.
    """
    f1 = F1Score().score(result.answer or "", [case.expected_answer])
    if f1 < threshold:
        return True
    return result.trace is not None and result.trace.grounding_decision == "ungrounded"


def _retrieval_metrics(
    case: BenchmarkCase, result: QueryResult
) -> tuple[float | None, float | None]:
    """Per-case retrieval recall+precision against `expected_source_ids`.

    Returns `(None, None)` when the case omits expected IDs. Recall is
    1.0 when at least one expected ID appears among the result's sources,
    0.0 otherwise. Precision is the fraction of result sources whose
    `source_id` is in `expected_source_ids`.
    """
    if case.expected_source_ids is None:
        return None, None

    expected = set(case.expected_source_ids)
    retrieved_ids = [s.source_id for s in result.sources]

    if not retrieved_ids:
        return 0.0, 0.0

    hits = sum(1 for rid in retrieved_ids if rid in expected)
    recall = 1.0 if hits > 0 else 0.0
    precision = hits / len(retrieved_ids)
    return recall, precision


async def run_benchmark(
    cases: list[BenchmarkCase],
    query_fn: _QueryFn,
    config: BenchmarkConfig | None = None,
    llm_judge: LLMJudgment | None = None,
) -> BenchmarkReport:
    """Execute `cases` against `query_fn` and aggregate a `BenchmarkReport`.

    `query_fn` is the engine's bound `query` method (or a stub in tests);
    it must accept `(query: str, *, trace: bool)` and return a
    `QueryResult` whose `trace` field is populated when `trace=True`.

    Concurrency is bounded by `config.concurrency` via `run_concurrent`
    — at most that many `query_fn` invocations run in parallel.
    """
    cfg = config or BenchmarkConfig()

    logger.info(
        "benchmark starting: %d cases, concurrency=%d", len(cases), cfg.concurrency
    )

    em_metric = ExactMatch()
    f1_metric = F1Score()

    async def _run_one(case: BenchmarkCase) -> BenchmarkCaseResult:
        result = await query_fn(case.query, trace=True)
        em = em_metric.score(result.answer or "", [case.expected_answer])
        f1 = f1_metric.score(result.answer or "", [case.expected_answer])
        recall, precision = _retrieval_metrics(case, result)
        per_case: dict[str, float] = {"em": em, "f1": f1}
        if recall is not None and precision is not None:
            per_case["retrieval_recall"] = recall
            per_case["retrieval_precision"] = precision
        if llm_judge is not None:
            judge_score = await llm_judge.score(
                result.answer or "", [case.expected_answer], query=case.query
            )
            per_case["llm_judge"] = judge_score

        failure: FailureClassification | None = None
        if _is_failure(case, result, cfg.failure_threshold) and result.trace is not None:
            failure = classify_failure(case.query, result.trace)

        return BenchmarkCaseResult(case=case, result=result, failure=failure, metrics=per_case)

    per_case_results = await run_concurrent(cases, _run_one, concurrency=cfg.concurrency)

    n = len(per_case_results)
    em_total = sum(r.metrics.get("em", 0.0) for r in per_case_results)
    f1_total = sum(r.metrics.get("f1", 0.0) for r in per_case_results)
    generation_em = em_total / n if n else 0.0
    generation_f1 = f1_total / n if n else 0.0

    all_have_expected = bool(cases) and all(c.expected_source_ids is not None for c in cases)
    if all_have_expected:
        recall_total = sum(r.metrics.get("retrieval_recall", 0.0) for r in per_case_results)
        precision_total = sum(r.metrics.get("retrieval_precision", 0.0) for r in per_case_results)
        retrieval_recall: float | None = recall_total / n if n else 0.0
        retrieval_precision: float | None = precision_total / n if n else 0.0
    else:
        retrieval_recall = None
        retrieval_precision = None

    if llm_judge is not None and n:
        judge_total = sum(r.metrics.get("llm_judge", 0.0) for r in per_case_results)
        llm_judge_score: float | None = judge_total / n
    else:
        llm_judge_score = None

    distribution: Counter[str] = Counter()
    for r in per_case_results:
        if r.failure is not None:
            distribution[r.failure.type.name] += 1

    logger.info(
        "benchmark complete: total=%d em=%.3f f1=%.3f failures=%d",
        n,
        generation_em,
        generation_f1,
        sum(distribution.values()),
    )

    return BenchmarkReport(
        total_cases=n,
        retrieval_recall=retrieval_recall,
        retrieval_precision=retrieval_precision,
        generation_em=generation_em,
        generation_f1=generation_f1,
        llm_judge_score=llm_judge_score,
        failure_distribution=dict(distribution),
        per_case_results=list(per_case_results),
    )


