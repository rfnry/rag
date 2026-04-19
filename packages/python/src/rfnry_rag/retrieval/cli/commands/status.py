from __future__ import annotations

import click

from rfnry_rag.retrieval.cli import cli, run_async
from rfnry_rag.retrieval.cli.config import ConfigError, load_config
from rfnry_rag.retrieval.cli.output import OutputMode, print_error, print_json


@cli.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Validate config and test connections."""
    mode: OutputMode = ctx.obj["output_mode"]
    config_path = ctx.obj["config_path"]

    try:
        server_config = load_config(config_path)
    except ConfigError as e:
        print_error(str(e), mode)
        raise SystemExit(1) from None

    if mode == OutputMode.PRETTY:
        click.echo("Config: valid")

    run_async(_test_connection(server_config, mode))


async def _test_connection(server_config, mode: OutputMode) -> None:
    from rfnry_rag.retrieval.server import RagEngine

    try:
        async with RagEngine(server_config) as rag:
            sources = await rag.knowledge.list()
            if mode == OutputMode.PRETTY:
                click.echo("Qdrant: connected")
                click.echo(f"Sources: {len(sources)}")
                click.echo("\nReady.")
            else:
                print_json(
                    {
                        "status": "ok",
                        "sources": len(sources),
                    }
                )
    except Exception as e:
        print_error(f"Connection failed: {e}", mode)
        raise SystemExit(1) from None
