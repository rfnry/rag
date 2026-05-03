from __future__ import annotations

from dataclasses import asdict

import click

from rfnry_knowledge.cli import cli, run_async
from rfnry_knowledge.cli.config import ConfigError, load_config
from rfnry_knowledge.cli.output import OutputMode, print_error, print_json, print_retrieved_chunks


@cli.command()
@click.argument("text")
@click.option("-k", "--knowledge-id", default=None, help="Knowledge namespace.")
@click.option("--min-score", type=float, default=None, help="Minimum retrieval score (0.0-1.0).")
@click.pass_context
def retrieve(ctx: click.Context, text: str, knowledge_id: str | None, min_score: float | None) -> None:
    """Retrieve chunks from the knowledge base (no LLM generation)."""
    mode: OutputMode = ctx.obj["output_mode"]
    config_path = ctx.obj["config_path"]

    try:
        server_config = load_config(config_path)
    except ConfigError as e:
        print_error(str(e), mode)
        raise SystemExit(1) from None

    run_async(_retrieve(server_config, text, knowledge_id, min_score, mode))


async def _retrieve(server_config, text, knowledge_id, min_score, mode):
    from rfnry_knowledge.knowledge.engine import KnowledgeEngine

    try:
        async with KnowledgeEngine(server_config) as engine:
            chunks, _ = await engine.retrieve(text, knowledge_id=knowledge_id, min_score=min_score)
            if mode == OutputMode.JSON:
                print_json({"chunks": [asdict(c) for c in chunks]})
            else:
                print_retrieved_chunks(chunks)
    except Exception as e:
        print_error(str(e), mode)
        raise SystemExit(1) from None
