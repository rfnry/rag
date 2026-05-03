from __future__ import annotations

from dataclasses import asdict

import click

from rfnry_knowledge.cli import cli, run_async
from rfnry_knowledge.cli.config import ConfigError, load_config
from rfnry_knowledge.cli.output import (
    OutputMode,
    print_chunks,
    print_error,
    print_health,
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
    from rfnry_knowledge.knowledge.engine import KnowledgeEngine

    try:
        async with KnowledgeEngine(server_config) as engine:
            sources = await engine.knowledge.list(knowledge_id=knowledge_id)
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
    from rfnry_knowledge.knowledge.engine import KnowledgeEngine

    try:
        async with KnowledgeEngine(server_config) as engine:
            source = await engine.knowledge.get(source_id)
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
    from rfnry_knowledge.knowledge.engine import KnowledgeEngine

    try:
        async with KnowledgeEngine(server_config) as engine:
            chunks = await engine.knowledge.get_chunks(source_id)
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
    from rfnry_knowledge.knowledge.engine import KnowledgeEngine

    try:
        async with KnowledgeEngine(server_config) as engine:
            stats = await engine.knowledge.get_stats(source_id)
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


@knowledge.command("inspect")
@click.argument("source_id")
@click.pass_context
def inspect_source(ctx: click.Context, source_id: str) -> None:
    """Show ingestion notes, embedding freshness, and retrieval stats for a source."""
    mode: OutputMode = ctx.obj["output_mode"]
    config_path = ctx.obj["config_path"]

    try:
        server_config = load_config(config_path)
    except ConfigError as e:
        print_error(str(e), mode)
        raise SystemExit(1) from None

    run_async(_inspect(server_config, source_id, mode))


async def _inspect(server_config, source_id, mode):
    from rfnry_knowledge.knowledge.engine import KnowledgeEngine

    try:
        async with KnowledgeEngine(server_config) as engine:
            health = await engine.knowledge.health(source_id)
            if health is None:
                print_error(f"Source not found: {source_id}", mode)
                raise SystemExit(1) from None
            if mode == OutputMode.JSON:
                print_json(asdict(health))
            else:
                print_health(health)
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
    from rfnry_knowledge.knowledge.engine import KnowledgeEngine

    try:
        async with KnowledgeEngine(server_config) as engine:
            deleted = await engine.knowledge.remove(source_id)
            data = {"source_id": source_id, "deleted_vectors": deleted}
            print_success(
                f"Removed {source_id}: {deleted} vectors deleted",
                data,
                mode,
            )
    except Exception as e:
        print_error(str(e), mode)
        raise SystemExit(1) from None
