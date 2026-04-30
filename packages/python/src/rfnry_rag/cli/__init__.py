from __future__ import annotations

import asyncio

import click

from rfnry_rag.cli.output import get_output_mode


@click.group()
@click.option("--json", "output_format", flag_value="json", help="Force JSON output.")
@click.option("--pretty", "output_format", flag_value="pretty", help="Force human-readable output.")
@click.option("--config", "config_path", type=click.Path(), default=None, help="Override config file path.")
@click.pass_context
def cli(ctx: click.Context, output_format: str | None, config_path: str | None) -> None:
    """rfnry-rag — CLI for the rfnry-rag SDK."""
    ctx.ensure_object(dict)
    ctx.obj["output_mode"] = get_output_mode(output_format)
    ctx.obj["config_path"] = config_path


def run_async(coro):
    """Run an async coroutine from a sync click command."""
    try:
        asyncio.run(coro)
    except KeyboardInterrupt:
        raise SystemExit(130) from None


def main() -> None:
    import rfnry_rag.cli.commands.benchmark  # noqa: F401
    import rfnry_rag.cli.commands.ingest  # noqa: F401
    import rfnry_rag.cli.commands.init  # noqa: F401
    import rfnry_rag.cli.commands.knowledge  # noqa: F401
    import rfnry_rag.cli.commands.query  # noqa: F401
    import rfnry_rag.cli.commands.retrieve  # noqa: F401
    import rfnry_rag.cli.commands.session  # noqa: F401
    import rfnry_rag.cli.commands.status  # noqa: F401

    cli()
