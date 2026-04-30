"""CLI smoke test for `rfnry-rag benchmark`.

Stubs `RagEngine` (the async context manager) and `load_config` so the
test exercises the command's plumbing — argument parsing, JSON loading,
report formatting, exit code — without standing up real stores. Asserts
exit code 0 and that the summary appears in stdout.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from click.testing import CliRunner

from rfnry_rag.retrieval.cli import cli
from rfnry_rag.retrieval.modules.evaluation.benchmark import BenchmarkReport


def test_benchmark_command_invokes_engine_and_prints_summary(tmp_path: Path) -> None:
    cases_file = tmp_path / "cases.json"
    cases_file.write_text(
        json.dumps(
            [
                {"query": "q1", "expected_answer": "a1"},
                {"query": "q2", "expected_answer": "a2"},
            ]
        )
    )

    fake_report = BenchmarkReport(
        total_cases=2,
        retrieval_recall=None,
        retrieval_precision=None,
        generation_em=0.5,
        generation_f1=0.75,
        llm_judge_score=None,
        failure_count=1,
        per_case_results=[],
    )

    fake_engine = MagicMock()
    fake_engine.benchmark = AsyncMock(return_value=fake_report)
    fake_engine.__aenter__ = AsyncMock(return_value=fake_engine)
    fake_engine.__aexit__ = AsyncMock(return_value=None)

    # `import rfnry_rag.retrieval.cli.commands.benchmark` registers the
    # `benchmark` subcommand on the shared `cli` group; once imported, the
    # CliRunner can invoke it by name.
    import rfnry_rag.retrieval.cli.commands.benchmark  # noqa: F401

    with (
        patch(
            "rfnry_rag.retrieval.cli.commands.benchmark.load_config",
            return_value=SimpleNamespace(),
        ),
        patch(
            "rfnry_rag.retrieval.server.RagEngine",
            return_value=fake_engine,
        ),
    ):
        runner = CliRunner()
        # `--pretty` forces human-readable output: CliRunner has no TTY, so
        # `get_output_mode` would otherwise default to JSON (the production
        # behavior when stdout is piped).
        result = runner.invoke(cli, ["--pretty", "benchmark", str(cases_file), "-k", "kb-1"])

    assert result.exit_code == 0, result.output
    assert "Benchmark report" in result.output
    assert "EM:" in result.output
    assert "F1:" in result.output
    assert "Failures: 1" in result.output
