from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import click

from rfnry_rag.reasoning.cli import cli, run_async
from rfnry_rag.reasoning.cli.config import build_analysis_service, load_config
from rfnry_rag.reasoning.cli.constants import ConfigError
from rfnry_rag.reasoning.cli.output import OutputMode, print_analysis, print_error, print_json


@cli.command("analyze-context")
@click.option("--file", "file_path", type=click.Path(exists=True), help="JSON file with messages array.")
@click.option("--summarize", is_flag=True, help="Include summary in output.")
@click.option(
    "--dimensions", "dimensions_file", type=click.Path(exists=True), help="JSON file with dimension definitions."
)
@click.pass_context
def analyze_context(ctx: click.Context, file_path: str | None, summarize: bool, dimensions_file: str | None) -> None:
    """Analyze a multi-segment context (conversation, email chain, chapters)."""
    mode: OutputMode = ctx.obj["output_mode"]

    if file_path:
        raw_json = Path(file_path).read_text()
    elif not sys.stdin.isatty():
        raw_json = sys.stdin.read()
    else:
        print_error("Provide --file or pipe JSON stdin.", mode)
        raise SystemExit(1) from None

    try:
        messages_data = json.loads(raw_json)
    except json.JSONDecodeError as e:
        print_error(f"Invalid JSON: {e}", mode)
        raise SystemExit(1) from None

    if not isinstance(messages_data, list) or not messages_data:
        print_error("Input must be a non-empty JSON array of {role, text} objects.", mode)
        raise SystemExit(1) from None

    try:
        toml = load_config(ctx.obj["config_path"])
    except ConfigError as e:
        print_error(str(e), mode)
        raise SystemExit(1) from None

    run_async(_analyze_context(toml, messages_data, summarize, dimensions_file, mode))


async def _analyze_context(
    toml: dict[str, Any],
    messages_data: list[dict[str, str]],
    summarize: bool,
    dimensions_file: str | None,
    mode: OutputMode,
) -> None:
    from rfnry_rag.reasoning.modules.analysis.models import (
        AnalysisConfig,
        ContextTrackingConfig,
        DimensionDefinition,
        Message,
    )

    try:
        service = build_analysis_service(toml)

        messages = [Message(text=m["text"], role=m["role"]) for m in messages_data]

        config = AnalysisConfig(
            summarize=summarize,
            context_tracking=ContextTrackingConfig(),
        )
        if dimensions_file:
            raw = json.loads(Path(dimensions_file).read_text())
            config.dimensions = [DimensionDefinition(**d) for d in raw]

        result = await service.analyze_context(messages, config=config)

        if mode == OutputMode.JSON:
            print_json(result)
        else:
            print_analysis(result)
    except Exception as e:
        print_error(str(e), mode)
        raise SystemExit(1) from None
