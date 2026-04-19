from __future__ import annotations

import click

from rfnry_rag.retrieval.cli import cli, run_async
from rfnry_rag.retrieval.cli.config import ConfigError, load_config
from rfnry_rag.retrieval.cli.output import OutputMode, print_error, print_json, print_query_result


@cli.command()
@click.argument("text")
@click.option("-k", "--knowledge-id", default=None, help="Knowledge namespace.")
@click.option("--min-score", type=float, default=None, help="Minimum retrieval score (0.0-1.0).")
@click.option("--session", "session_name", default=None, help="Session name for multi-turn context.")
@click.pass_context
def query(
    ctx: click.Context, text: str, knowledge_id: str | None, min_score: float | None, session_name: str | None
) -> None:
    """Query the knowledge base (retrieve + generate answer)."""
    mode: OutputMode = ctx.obj["output_mode"]
    config_path = ctx.obj["config_path"]

    try:
        server_config = load_config(config_path)
    except ConfigError as e:
        print_error(str(e), mode)
        raise SystemExit(1) from None

    run_async(_query(server_config, text, knowledge_id, min_score, session_name, mode))


async def _query(server_config, text, knowledge_id, min_score, session_name, mode):
    from rfnry_rag.retrieval.cli.commands.session import load_session, save_turn
    from rfnry_rag.retrieval.server import RagEngine

    history = load_session(session_name) if session_name else None

    try:
        async with RagEngine(server_config) as rag:
            result = await rag.query(text, knowledge_id=knowledge_id, history=history, min_score=min_score)
            if session_name and result.answer:
                save_turn(session_name, text, result.answer)

            if mode == OutputMode.JSON:
                print_json(result)
            else:
                print_query_result(result)
    except RuntimeError as e:
        if "generation" in str(e).lower():
            print_error("Generation not configured. Add [generation] to config.toml.", mode)
        else:
            print_error(str(e), mode)
        raise SystemExit(1) from None
    except Exception as e:
        print_error(str(e), mode)
        raise SystemExit(1) from None
