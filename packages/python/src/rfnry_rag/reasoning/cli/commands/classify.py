from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, cast

import click

from rfnry_rag.reasoning.cli import cli, resolve_input, run_async
from rfnry_rag.reasoning.cli.config import build_classification_service, load_config
from rfnry_rag.reasoning.cli.constants import ConfigError
from rfnry_rag.reasoning.cli.output import OutputMode, print_classification, print_error, print_json


@cli.command()
@click.argument("text", required=False)
@click.option("--file", "file_path", type=click.Path(exists=True), help="Read text from file.")
@click.option(
    "--categories",
    "categories_file",
    type=click.Path(exists=True),
    required=True,
    help="JSON file with category definitions.",
)
@click.option("--strategy", type=click.Choice(["llm", "hybrid"]), default="llm", help="Classification strategy.")
@click.pass_context
def classify(ctx: click.Context, text: str | None, file_path: str | None, categories_file: str, strategy: str) -> None:
    """Classify text into categories."""
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

    run_async(_classify(toml, resolved, categories_file, strategy, mode))


async def _classify(toml: dict[str, Any], text: str, categories_file: str, strategy: str, mode: OutputMode) -> None:
    from rfnry_rag.reasoning.modules.classification.models import CategoryDefinition, ClassificationConfig

    try:
        service = build_classification_service(toml)

        raw = json.loads(Path(categories_file).read_text())
        categories = [CategoryDefinition(**c) for c in raw]
        config = ClassificationConfig(strategy=cast(Literal["llm", "hybrid"], strategy))

        result = await service.classify(text, categories, config=config)

        if mode == OutputMode.JSON:
            print_json(result)
        else:
            print_classification(result)
    except Exception as e:
        print_error(str(e), mode)
        raise SystemExit(1) from None
