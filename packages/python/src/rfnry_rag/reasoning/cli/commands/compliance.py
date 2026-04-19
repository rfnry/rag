from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click

from rfnry_rag.reasoning.cli import cli, resolve_input, resolve_references, run_async
from rfnry_rag.reasoning.cli.config import build_compliance_service, load_config
from rfnry_rag.reasoning.cli.constants import ConfigError
from rfnry_rag.reasoning.cli.output import OutputMode, print_compliance, print_error, print_json


@cli.command()
@click.argument("text", required=False)
@click.option("--file", "file_path", type=click.Path(exists=True), help="Read text from file.")
@click.option(
    "--references",
    "reference_paths",
    type=click.Path(),
    required=True,
    multiple=True,
    help="Reference document(s) or folder to check against.",
)
@click.option(
    "--dimensions",
    "dimensions_file",
    type=click.Path(exists=True),
    help="JSON file with compliance dimension definitions.",
)
@click.option("--threshold", type=float, default=None, help="Score threshold for compliant gate (0.0-1.0).")
@click.pass_context
def compliance(
    ctx: click.Context,
    text: str | None,
    file_path: str | None,
    reference_paths: tuple[str, ...],
    dimensions_file: str | None,
    threshold: float | None,
) -> None:
    """Check text compliance against reference document(s)."""
    mode: OutputMode = ctx.obj["output_mode"]

    try:
        resolved = resolve_input(text, file_path)
    except click.UsageError:
        print_error("Provide text, --file, or pipe stdin.", mode)
        raise SystemExit(1) from None

    try:
        reference = resolve_references(reference_paths)
    except click.UsageError as e:
        print_error(str(e), mode)
        raise SystemExit(1) from None

    try:
        toml = load_config(ctx.obj["config_path"])
    except ConfigError as e:
        print_error(str(e), mode)
        raise SystemExit(1) from None

    run_async(_compliance(toml, resolved, reference, dimensions_file, threshold, mode))


async def _compliance(
    toml: dict[str, Any],
    text: str,
    reference: str,
    dimensions_file: str | None,
    threshold: float | None,
    mode: OutputMode,
) -> None:
    from rfnry_rag.reasoning.modules.compliance.models import ComplianceConfig, ComplianceDimensionDefinition

    try:
        service = build_compliance_service(toml)

        config = ComplianceConfig(threshold=threshold)
        if dimensions_file:
            raw = json.loads(Path(dimensions_file).read_text())
            config.dimensions = [ComplianceDimensionDefinition(**d) for d in raw]

        result = await service.check(text, reference, config=config)

        if mode == OutputMode.JSON:
            print_json(result)
        else:
            print_compliance(result)
    except Exception as e:
        print_error(str(e), mode)
        raise SystemExit(1) from None
