from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, cast

import click

from rfnry_rag.reasoning.cli import cli, run_async
from rfnry_rag.reasoning.cli.config import build_evaluation_service, load_config
from rfnry_rag.reasoning.cli.constants import ConfigError
from rfnry_rag.reasoning.cli.output import OutputMode, print_error, print_evaluation, print_json


@cli.command()
@click.option(
    "--generated", "generated_file", type=click.Path(exists=True), required=True, help="File with generated text."
)
@click.option(
    "--reference", "reference_file", type=click.Path(exists=True), required=True, help="File with reference text."
)
@click.option(
    "--strategy", type=click.Choice(["similarity", "judge", "combined"]), default="judge", help="Evaluation strategy."
)
@click.pass_context
def evaluate(ctx: click.Context, generated_file: str, reference_file: str, strategy: str) -> None:
    """Evaluate generated text against a reference."""
    mode: OutputMode = ctx.obj["output_mode"]

    try:
        toml = load_config(ctx.obj["config_path"])
    except ConfigError as e:
        print_error(str(e), mode)
        raise SystemExit(1) from None

    generated = Path(generated_file).read_text()
    reference = Path(reference_file).read_text()

    run_async(_evaluate(toml, generated, reference, strategy, mode))


async def _evaluate(toml: dict[str, Any], generated: str, reference: str, strategy: str, mode: OutputMode) -> None:
    from rfnry_rag.reasoning.modules.evaluation.models import EvaluationConfig, EvaluationPair

    try:
        service = build_evaluation_service(toml)
        pair = EvaluationPair(generated=generated, reference=reference)
        config = EvaluationConfig(strategy=cast(Literal["similarity", "judge", "combined"], strategy))

        result = await service.evaluate(pair, config=config)

        if mode == OutputMode.JSON:
            print_json(result)
        else:
            print_evaluation(result)
    except Exception as e:
        print_error(str(e), mode)
        raise SystemExit(1) from None
