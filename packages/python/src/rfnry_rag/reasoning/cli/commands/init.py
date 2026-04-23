from __future__ import annotations

import click

from rfnry_rag.reasoning.cli import cli
from rfnry_rag.reasoning.cli.constants import CONFIG_DIR, CONFIG_FILE, ENV_FILE

_CONFIG_TEMPLATE = """\
[language_model]
provider = "anthropic"
model = "claude-sonnet-4-20250514"
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
        # chmod 0o600 to match retrieval init — owner-only read even though the
        # reasoning template currently holds no secrets, since provider/model
        # identifiers can aid reconnaissance and future templates may add keys.
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
        click.echo("\nEdit these files, then run 'rfnry-rag reasoning status' to verify.")
    else:
        click.echo("\nAlready initialized. Edit files or run 'rfnry-rag reasoning status'.")
