from __future__ import annotations

import os
from typing import Any

import click

from rfnry_rag.reasoning.cli import cli, run_async
from rfnry_rag.reasoning.cli.config import build_lm_client, load_config
from rfnry_rag.reasoning.cli.constants import CONFIG_FILE, ENV_FILE, ConfigError
from rfnry_rag.reasoning.cli.output import OutputMode, print_error, print_json
from rfnry_rag.reasoning.common.language_model import LanguageModelClient


@cli.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Validate config and test LLM connection."""
    mode: OutputMode = ctx.obj["output_mode"]
    config_path = ctx.obj["config_path"]

    try:
        toml = load_config(config_path)
    except ConfigError as e:
        print_error(str(e), mode)
        raise SystemExit(1) from None

    if mode == OutputMode.PRETTY:
        path = config_path or str(CONFIG_FILE)
        click.echo(f"Config: {path}")
        if os.path.exists(ENV_FILE):
            click.echo(f".env: {ENV_FILE}")
        else:
            click.echo(".env: not found (API keys must be in environment)")

    try:
        lm_client = build_lm_client(toml)
    except ConfigError as e:
        print_error(str(e), mode)
        raise SystemExit(1) from None

    if mode == OutputMode.PRETTY:
        click.echo(f"Provider: {lm_client.provider.provider}")
        click.echo(f"Model: {lm_client.provider.model}")
        if lm_client.fallback:
            click.echo(f"Fallback: {lm_client.fallback.provider}/{lm_client.fallback.model}")
        if toml.get("embeddings"):
            emb = toml["embeddings"]
            click.echo(f"Embeddings: {emb.get('provider', 'openai')}/{emb.get('model', 'default')}")
        if toml.get("vector_store"):
            vs = toml["vector_store"]
            click.echo(f"Vector store: {vs.get('provider', 'qdrant')} @ {vs.get('url', 'localhost')}")

    run_async(_test_connection(lm_client, toml, mode))


async def _test_connection(lm_client: LanguageModelClient, toml: dict[str, Any], mode: OutputMode) -> None:
    from rfnry_rag.reasoning.modules.analysis.service import AnalysisService

    try:
        service = AnalysisService(lm_client=lm_client)
        await service.analyze("test", config=None)
        if mode == OutputMode.PRETTY:
            click.echo("LLM: connected")
            click.echo("\nReady.")
        else:
            status_data: dict[str, Any] = {
                "status": "ok",
                "provider": lm_client.provider.provider,
                "model": lm_client.provider.model,
            }
            if toml.get("embeddings"):
                status_data["embeddings"] = toml["embeddings"].get("provider", "openai")
            if toml.get("vector_store"):
                status_data["vector_store"] = toml["vector_store"].get("provider", "qdrant")
            print_json(status_data)
    except Exception as e:
        print_error(f"LLM connection failed: {e}", mode)
        raise SystemExit(1) from None
