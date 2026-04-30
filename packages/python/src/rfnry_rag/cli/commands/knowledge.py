from __future__ import annotations

from dataclasses import asdict

import click

from rfnry_rag.cli import cli, run_async
from rfnry_rag.cli.config import ConfigError, load_config
from rfnry_rag.cli.output import (
    OutputMode,
    print_chunks,
    print_error,
    print_json,
    print_source,
    print_source_list,
    print_stats,
    print_success,
)


@cli.group()
def knowledge() -> None:
    """Manage knowledge sources."""
    pass


@knowledge.command("list")
@click.option("-k", "--knowledge-id", default=None, help="Filter by knowledge namespace.")
@click.pass_context
def list_sources(ctx: click.Context, knowledge_id: str | None) -> None:
    """List all sources."""
    mode: OutputMode = ctx.obj["output_mode"]
    config_path = ctx.obj["config_path"]

    try:
        server_config = load_config(config_path)
    except ConfigError as e:
        print_error(str(e), mode)
        raise SystemExit(1) from None

    run_async(_list(server_config, knowledge_id, mode))


async def _list(server_config, knowledge_id, mode):
    from rfnry_rag.server import RagEngine

    try:
        async with RagEngine(server_config) as rag:
            sources = await rag.knowledge.list(knowledge_id=knowledge_id)
            if mode == OutputMode.JSON:
                print_json({"sources": [asdict(s) for s in sources]})
            else:
                print_source_list(sources)
    except Exception as e:
        print_error(str(e), mode)
        raise SystemExit(1) from None


@knowledge.command("get")
@click.argument("source_id")
@click.pass_context
def get_source(ctx: click.Context, source_id: str) -> None:
    """Get source details."""
    mode: OutputMode = ctx.obj["output_mode"]
    config_path = ctx.obj["config_path"]

    try:
        server_config = load_config(config_path)
    except ConfigError as e:
        print_error(str(e), mode)
        raise SystemExit(1) from None

    run_async(_get(server_config, source_id, mode))


async def _get(server_config, source_id, mode):
    from rfnry_rag.server import RagEngine

    try:
        async with RagEngine(server_config) as rag:
            source = await rag.knowledge.get(source_id)
            if source is None:
                print_error(f"Source not found: {source_id}", mode)
                raise SystemExit(1) from None
            if mode == OutputMode.JSON:
                print_json(asdict(source))
            else:
                print_source(source)
    except SystemExit:
        raise
    except Exception as e:
        print_error(str(e), mode)
        raise SystemExit(1) from None


@knowledge.command("chunks")
@click.argument("source_id")
@click.pass_context
def get_chunks(ctx: click.Context, source_id: str) -> None:
    """Inspect chunks belonging to a source."""
    mode: OutputMode = ctx.obj["output_mode"]
    config_path = ctx.obj["config_path"]

    try:
        server_config = load_config(config_path)
    except ConfigError as e:
        print_error(str(e), mode)
        raise SystemExit(1) from None

    run_async(_chunks(server_config, source_id, mode))


async def _chunks(server_config, source_id, mode):
    from rfnry_rag.server import RagEngine

    try:
        async with RagEngine(server_config) as rag:
            chunks = await rag.knowledge.get_chunks(source_id)
            if mode == OutputMode.JSON:
                print_json({"chunks": [asdict(c) for c in chunks]})
            else:
                print_chunks(chunks)
    except Exception as e:
        print_error(str(e), mode)
        raise SystemExit(1) from None


@knowledge.command("stats")
@click.argument("source_id")
@click.pass_context
def get_stats(ctx: click.Context, source_id: str) -> None:
    """Show hit statistics for a source."""
    mode: OutputMode = ctx.obj["output_mode"]
    config_path = ctx.obj["config_path"]

    try:
        server_config = load_config(config_path)
    except ConfigError as e:
        print_error(str(e), mode)
        raise SystemExit(1) from None

    run_async(_stats(server_config, source_id, mode))


async def _stats(server_config, source_id, mode):
    from rfnry_rag.server import RagEngine

    try:
        async with RagEngine(server_config) as rag:
            stats = await rag.knowledge.get_stats(source_id)
            if stats is None:
                print_error("Stats require metadata store. Add [persistence.metadata] to config.toml.", mode)
                raise SystemExit(1) from None
            if mode == OutputMode.JSON:
                print_json(asdict(stats))
            else:
                print_stats(stats)
    except SystemExit:
        raise
    except Exception as e:
        print_error(str(e), mode)
        raise SystemExit(1) from None


@knowledge.command("remove")
@click.argument("source_id")
@click.pass_context
def remove_source(ctx: click.Context, source_id: str) -> None:
    """Delete a source and all its chunks."""
    mode: OutputMode = ctx.obj["output_mode"]
    config_path = ctx.obj["config_path"]

    try:
        server_config = load_config(config_path)
    except ConfigError as e:
        print_error(str(e), mode)
        raise SystemExit(1) from None

    run_async(_remove(server_config, source_id, mode))


async def _remove(server_config, source_id, mode):
    from rfnry_rag.server import RagEngine

    try:
        async with RagEngine(server_config) as rag:
            deleted = await rag.knowledge.remove(source_id)
            data = {"source_id": source_id, "deleted_vectors": deleted}
            print_success(
                f"Removed {source_id}: {deleted} vectors deleted",
                data,
                mode,
            )
    except Exception as e:
        print_error(str(e), mode)
        raise SystemExit(1) from None
