from __future__ import annotations

import click

from rfnry_knowledge.cli import cli
from rfnry_knowledge.cli.constants import CONFIG_DIR, CONFIG_FILE, ENV_FILE

_CONFIG_TEMPLATE = """\
[persistence]
vector_store = "qdrant"
url = "http://localhost:6333"
collection = "docs"

[ingestion]
embeddings = "openai"
model = "text-embedding-3-small"
"""

_ENV_TEMPLATE = """\
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
COHERE_API_KEY=
VOYAGE_API_KEY=
"""


@cli.command()
def init() -> None:
    """Create config and .env templates."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    created = []

    if CONFIG_FILE.exists():
        click.echo(f"Config already exists: {CONFIG_FILE}")
    else:
        CONFIG_FILE.write_text(_CONFIG_TEMPLATE)
        # config.toml can contain PostgreSQL URLs with embedded credentials;
        # restrict to user-only read/write like the sibling .env file.
        CONFIG_FILE.chmod(0o600)
        created.append(str(CONFIG_FILE))

    if ENV_FILE.exists():
        click.echo(f".env already exists: {ENV_FILE}")
    else:
        ENV_FILE.write_text(_ENV_TEMPLATE)
        ENV_FILE.chmod(0o600)
        created.append(str(ENV_FILE))

    if created:
        click.echo("\nCreated:")
        for path in created:
            click.echo(f"  {path}")
        click.echo("\nEdit these files, then run 'rfnry-knowledge retrieval status' to verify.")
    else:
        click.echo("\nAlready initialized. Edit files or run 'rfnry-knowledge retrieval status'.")
