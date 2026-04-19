from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click

from rfnry_rag.reasoning.cli import cli, resolve_input, run_async
from rfnry_rag.reasoning.cli.config import build_analysis_service, load_config
from rfnry_rag.reasoning.cli.constants import ConfigError
from rfnry_rag.reasoning.cli.output import OutputMode, print_analysis, print_error, print_json


@cli.command()
@click.argument("text", required=False)
@click.option("--file", "file_path", type=click.Path(exists=True), help="Read text from file.")
@click.option("--summarize", is_flag=True, help="Include summary in output.")
@click.option(
    "--dimensions", "dimensions_file", type=click.Path(exists=True), help="JSON file with dimension definitions."
)
@click.pass_context
def analyze(
    ctx: click.Context, text: str | None, file_path: str | None, summarize: bool, dimensions_file: str | None
) -> None:
    """Analyze text for intent, entities, and dimensions."""
    mode: OutputMode = ctx.obj["output_mode"]

    try:
        resolved = resolve_input(text, file_path)
    except click.UsageError:
        print_error("Provide text, --file, or pipe stdin.", mode)
        raise SystemExit(1) from None

    try:
        toml = load_config(ctx.obj["config_path"])
    except ConfigError as e:
        print_error(str(e), mode)
        raise SystemExit(1) from None

    run_async(_analyze(toml, resolved, summarize, dimensions_file, mode))


async def _analyze(
    toml: dict[str, Any], text: str, summarize: bool, dimensions_file: str | None, mode: OutputMode
) -> None:
    from rfnry_rag.reasoning.modules.analysis.models import AnalysisConfig, DimensionDefinition

    try:
        service = build_analysis_service(toml)

        config = AnalysisConfig(summarize=summarize)
        if dimensions_file:
            raw = json.loads(Path(dimensions_file).read_text())
            config.dimensions = [DimensionDefinition(**d) for d in raw]

        result = await service.analyze(text, config=config)

        if mode == OutputMode.JSON:
            print_json(result)
        else:
            print_analysis(result)
    except Exception as e:
        print_error(str(e), mode)
        raise SystemExit(1) from None
