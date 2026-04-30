"""Unified CLI entry point for rfnry-rag."""

from __future__ import annotations

import click

import rfnry_rag.retrieval.cli.commands.ingest
import rfnry_rag.retrieval.cli.commands.init
import rfnry_rag.retrieval.cli.commands.knowledge
import rfnry_rag.retrieval.cli.commands.query
import rfnry_rag.retrieval.cli.commands.retrieve
import rfnry_rag.retrieval.cli.commands.session
import rfnry_rag.retrieval.cli.commands.status
from rfnry_rag.retrieval.cli import cli as retrieval_cli


@click.group()
@click.version_option(package_name="rfnry_rag")
def main() -> None:
    """rfnry-rag — Retrieval toolkit."""


main.add_command(retrieval_cli, name="retrieval")


if __name__ == "__main__":
    main()
