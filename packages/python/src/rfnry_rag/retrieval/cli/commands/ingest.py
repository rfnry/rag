from __future__ import annotations

from pathlib import Path

import click

from rfnry_rag.retrieval.cli import cli, run_async
from rfnry_rag.retrieval.cli.config import ConfigError, load_config
from rfnry_rag.retrieval.cli.output import OutputMode, print_error, print_success


@cli.command()
@click.argument("file", required=False, type=click.Path())
@click.option("--text", "text_content", default=None, help="Ingest raw text instead of a file.")
@click.option("-k", "--knowledge-id", default=None, help="Knowledge namespace.")
@click.option("--source-type", default=None, help="Source type label.")
@click.option("--tree-index", is_flag=True, default=False, help="Build a tree index for structured retrieval.")
@click.pass_context
def ingest(
    ctx: click.Context,
    file: str | None,
    text_content: str | None,
    knowledge_id: str | None,
    source_type: str | None,
    tree_index: bool,
) -> None:
    """Ingest a file or text into the knowledge base."""
    mode: OutputMode = ctx.obj["output_mode"]
    config_path = ctx.obj["config_path"]

    if not file and not text_content:
        print_error("Provide a file path or --text 'content'", mode)
        raise SystemExit(1) from None
    if file and text_content:
        print_error("Provide either a file path or --text, not both.", mode)
        raise SystemExit(1) from None

    try:
        server_config = load_config(config_path)
    except ConfigError as e:
        print_error(str(e), mode)
        raise SystemExit(1) from None

    if file:
        run_async(_ingest_file(server_config, Path(file), knowledge_id, source_type, tree_index, mode))
    else:
        run_async(_ingest_text(server_config, text_content, knowledge_id, source_type, mode))


async def _ingest_file(server_config, file_path, knowledge_id, source_type, tree_index, mode):
    from rfnry_rag.retrieval.server import RagEngine

    try:
        async with RagEngine(server_config) as rag:
            source = await rag.ingest(
                file_path=file_path,
                knowledge_id=knowledge_id,
                source_type=source_type,
                tree_index=tree_index,
            )
            data = {
                "source_id": source.source_id,
                "chunk_count": source.chunk_count,
                "status": source.status,
            }
            print_success(
                f"Ingested: {source.source_id} ({source.chunk_count} chunks)",
                data,
                mode,
            )
    except Exception as e:
        print_error(str(e), mode)
        raise SystemExit(1) from None


async def _ingest_text(server_config, content, knowledge_id, source_type, mode):
    from rfnry_rag.retrieval.server import RagEngine

    try:
        async with RagEngine(server_config) as rag:
            source = await rag.ingest_text(
                content=content,
                knowledge_id=knowledge_id,
                source_type=source_type,
            )
            data = {
                "source_id": source.source_id,
                "chunk_count": source.chunk_count,
                "status": source.status,
            }
            print_success(
                f"Ingested: {source.source_id} ({source.chunk_count} chunks)",
                data,
                mode,
            )
    except Exception as e:
        print_error(str(e), mode)
        raise SystemExit(1) from None
