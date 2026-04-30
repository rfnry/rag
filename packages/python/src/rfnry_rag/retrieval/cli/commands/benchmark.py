from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import click

from rfnry_rag.retrieval.cli import cli, run_async
from rfnry_rag.retrieval.cli.config import ConfigError, load_config
from rfnry_rag.retrieval.cli.output import (
    OutputMode,
    print_benchmark_report,
    print_error,
    print_json,
)


@cli.command()
@click.argument("cases_file", type=click.Path(exists=True, dir_okay=False))
@click.option("-k", "--knowledge-id", default=None, help="Knowledge namespace.")
@click.option(
    "-o",
    "--output",
    "output_path",
    type=click.Path(dir_okay=False),
    default=None,
    help="Write full report (with per-case traces) as JSON.",
)
@click.option("-c", "--concurrency", type=int, default=None, help="Override BenchmarkConfig.concurrency.")
@click.option(
    "--failure-threshold",
    type=float,
    default=None,
    help="Override BenchmarkConfig.failure_threshold.",
)
@click.pass_context
def benchmark(
    ctx: click.Context,
    cases_file: str,
    knowledge_id: str | None,
    output_path: str | None,
    concurrency: int | None,
    failure_threshold: float | None,
) -> None:
    """Run a retrieval+generation benchmark over CASES_FILE (JSON list of cases)."""
    mode: OutputMode = ctx.obj["output_mode"]
    config_path = ctx.obj["config_path"]

    try:
        server_config = load_config(config_path)
    except ConfigError as e:
        print_error(str(e), mode)
        raise SystemExit(1) from None

    run_async(
        _benchmark(
            server_config,
            cases_file,
            knowledge_id,
            output_path,
            concurrency,
            failure_threshold,
            mode,
        )
    )


async def _benchmark(
    server_config,
    cases_file: str,
    knowledge_id: str | None,
    output_path: str | None,
    concurrency: int | None,
    failure_threshold: float | None,
    mode: OutputMode,
) -> None:
    from rfnry_rag.retrieval.modules.evaluation.benchmark import BenchmarkCase, BenchmarkConfig
    from rfnry_rag.retrieval.server import RagEngine

    try:
        raw = json.loads(Path(cases_file).read_text())
        cases = [BenchmarkCase(**c) for c in raw]

        bench_config: BenchmarkConfig | None
        if concurrency is not None or failure_threshold is not None:
            defaults = BenchmarkConfig()
            bench_config = BenchmarkConfig(
                concurrency=concurrency if concurrency is not None else defaults.concurrency,
                failure_threshold=failure_threshold if failure_threshold is not None else defaults.failure_threshold,
            )
        else:
            bench_config = None

        async with RagEngine(server_config) as rag:
            report = await rag.benchmark(cases, config=bench_config, knowledge_id=knowledge_id)

        if output_path:
            Path(output_path).write_text(json.dumps(asdict(report), indent=2, default=str))

        if mode == OutputMode.JSON:
            print_json(report)
        else:
            print_benchmark_report(report)
    except Exception as e:
        print_error(str(e), mode)
        raise SystemExit(1) from None
