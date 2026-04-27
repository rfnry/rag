"""R8.3 — Benchmark harness unit tests.

These exercise `run_benchmark` directly (the core orchestrator that
`RagEngine.benchmark` wraps) so we can stub `query_fn` without standing
up a full engine. The R8.1 + R8.2 plumbing is exercised end-to-end:
each case is run with `trace=True`, failures invoke `classify_failure`,
and the report aggregates by `FailureType.name`.

Bias-term hygiene: fixtures use generic identifiers (`q1`, `case_a`,
`kb-1`, `EntityXYZ`).
"""

from __future__ import annotations

import asyncio

import pytest

from rfnry_rag.common.errors import ConfigurationError
from rfnry_rag.retrieval.common.models import RetrievalTrace, RetrievedChunk
from rfnry_rag.retrieval.modules.evaluation import (
    BenchmarkCase,
    BenchmarkConfig,
    run_benchmark,
)
from rfnry_rag.retrieval.modules.generation.models import QueryResult, SourceReference


def _result(answer: str, *, sources: list[SourceReference] | None = None,
            trace: RetrievalTrace | None = None, grounded: bool = True) -> QueryResult:
    return QueryResult(
        answer=answer,
        sources=sources or [],
        grounded=grounded,
        confidence=0.9,
        trace=trace,
    )


def _src(source_id: str) -> SourceReference:
    return SourceReference(source_id=source_id, name=source_id, score=0.9)


def _chunk(chunk_id: str, score: float = 0.9) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=chunk_id, source_id="s1", content="text", score=score)


# 1. Aggregation -------------------------------------------------------------


async def test_benchmark_runs_all_cases_and_aggregates_em_f1() -> None:
    """5 cases, deterministic stub answers — EM/F1 are correctly averaged."""
    cases = [
        BenchmarkCase(query=f"q{i}", expected_answer=f"answer {i}")
        for i in range(5)
    ]

    async def query_fn(text: str, *, trace: bool) -> QueryResult:
        idx = int(text[1:])
        # Three of five exact match; remaining two are a partial overlap.
        if idx < 3:
            return _result(f"answer {idx}")
        return _result(f"different {idx}")

    report = await run_benchmark(cases, query_fn)

    assert report.total_cases == 5
    assert report.generation_em == pytest.approx(3 / 5)
    # The two non-exact answers each share token "{idx}" with their reference,
    # giving F1 = 2 * (1/2 * 1/2) / (1/2 + 1/2) = 0.5 each.
    assert report.generation_f1 == pytest.approx((3 * 1.0 + 2 * 0.5) / 5)


# 2. Retrieval recall when expected IDs supplied -----------------------------


async def test_benchmark_retrieval_recall_when_expected_ids_provided() -> None:
    """3 cases, IDs supplied; 2/3 retrievals contain a matching source — recall == 2/3."""
    cases = [
        BenchmarkCase(query="q1", expected_answer="a1", expected_source_ids=["src-1"]),
        BenchmarkCase(query="q2", expected_answer="a2", expected_source_ids=["src-2"]),
        BenchmarkCase(query="q3", expected_answer="a3", expected_source_ids=["src-3"]),
    ]

    async def query_fn(text: str, *, trace: bool) -> QueryResult:
        if text == "q1":
            return _result("a1", sources=[_src("src-1")])
        if text == "q2":
            return _result("a2", sources=[_src("src-2")])
        return _result("a3", sources=[_src("other")])

    report = await run_benchmark(cases, query_fn)

    assert report.retrieval_recall == pytest.approx(2 / 3)
    assert report.retrieval_precision is not None


# 3. Skip retrieval metrics when no expected IDs -----------------------------


async def test_benchmark_skips_retrieval_metrics_when_no_expected_ids() -> None:
    """At least one case lacks expected IDs — recall/precision report None, not 0.0."""
    cases = [
        BenchmarkCase(query="q1", expected_answer="a1"),
        BenchmarkCase(query="q2", expected_answer="a2"),
    ]

    async def query_fn(text: str, *, trace: bool) -> QueryResult:
        return _result("a1" if text == "q1" else "a2", sources=[_src("src-1")])

    report = await run_benchmark(cases, query_fn)

    assert report.retrieval_recall is None
    assert report.retrieval_precision is None


# 4. Failure classification via R8.2 -----------------------------------------


async def test_benchmark_classifies_failures_via_r8_2() -> None:
    """Failed cases route through classify_failure; histogram keys on FailureType.name."""
    cases = [
        BenchmarkCase(query="q1", expected_answer="expected one"),
        BenchmarkCase(query="q2", expected_answer="expected two"),
        BenchmarkCase(query="q3", expected_answer="expected three"),
    ]

    async def query_fn(text: str, *, trace: bool) -> QueryResult:
        if text == "q1":
            # VOCABULARY_MISMATCH: empty document channel + low vector top score.
            tr = RetrievalTrace(
                query=text,
                per_method_results={"document": [], "vector": [_chunk("c1", 0.21)]},
            )
            return _result("totally unrelated", trace=tr, grounded=False)
        if text == "q2":
            # CHUNK_BOUNDARY: high-score chunk but ungrounded.
            tr = RetrievalTrace(
                query=text,
                per_method_results={"vector": [_chunk("c1", 0.85)]},
                fused_results=[_chunk("c1", 0.85)],
                final_results=[_chunk("c1", 0.85)],
                grounding_decision="ungrounded",
            )
            return _result("partial", trace=tr, grounded=False)
        # SCOPE_MISS: every method empty + knowledge_id set.
        tr = RetrievalTrace(
            query=text,
            per_method_results={"document": [], "vector": []},
            knowledge_id="kb-1",
        )
        return _result("nothing", trace=tr, grounded=False)

    report = await run_benchmark(cases, query_fn)

    assert report.failure_distribution == {
        "VOCABULARY_MISMATCH": 1,
        "CHUNK_BOUNDARY": 1,
        "SCOPE_MISS": 1,
    }


# 5. Config bounds -----------------------------------------------------------


def test_benchmark_failure_threshold_config_bounded() -> None:
    """failure_threshold and concurrency reject out-of-range values."""
    with pytest.raises(ConfigurationError):
        BenchmarkConfig(failure_threshold=-0.1)
    with pytest.raises(ConfigurationError):
        BenchmarkConfig(failure_threshold=1.1)
    with pytest.raises(ConfigurationError):
        BenchmarkConfig(concurrency=0)
    with pytest.raises(ConfigurationError):
        BenchmarkConfig(concurrency=21)

    BenchmarkConfig(failure_threshold=0.0)
    BenchmarkConfig(failure_threshold=0.5)
    BenchmarkConfig(failure_threshold=1.0)
    BenchmarkConfig(concurrency=1)
    BenchmarkConfig(concurrency=20)


# 6. Concurrency bound -------------------------------------------------------


async def test_benchmark_concurrency_bounds_in_flight_calls() -> None:
    """Mirrors R3 pattern: lock-protected counter sees max in-flight <= concurrency, >= 2."""
    in_flight = 0
    max_in_flight = 0
    lock = asyncio.Lock()

    async def query_fn(text: str, *, trace: bool) -> QueryResult:
        nonlocal in_flight, max_in_flight
        async with lock:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
        await asyncio.sleep(0.01)
        async with lock:
            in_flight -= 1
        return _result("ok")

    cases = [BenchmarkCase(query=f"q{i}", expected_answer="ok") for i in range(10)]
    await run_benchmark(cases, query_fn, config=BenchmarkConfig(concurrency=3))

    assert max_in_flight <= 3
    assert max_in_flight >= 2
