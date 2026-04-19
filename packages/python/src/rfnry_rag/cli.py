"""Unified CLI entry point for rfnry-rag."""

from __future__ import annotations

import click

import rfnry_rag.reasoning.cli.commands.analyze
import rfnry_rag.reasoning.cli.commands.analyze_context
import rfnry_rag.reasoning.cli.commands.classify
import rfnry_rag.reasoning.cli.commands.compliance
import rfnry_rag.reasoning.cli.commands.evaluate
import rfnry_rag.reasoning.cli.commands.init
import rfnry_rag.reasoning.cli.commands.status
import rfnry_rag.retrieval.cli.commands.ingest
import rfnry_rag.retrieval.cli.commands.init
import rfnry_rag.retrieval.cli.commands.knowledge
import rfnry_rag.retrieval.cli.commands.query
import rfnry_rag.retrieval.cli.commands.retrieve
import rfnry_rag.retrieval.cli.commands.session
import rfnry_rag.retrieval.cli.commands.status
from rfnry_rag.reasoning.cli import cli as reasoning_cli
from rfnry_rag.retrieval.cli import cli as retrieval_cli


@click.group()
@click.version_option(package_name="rfnry_rag")
def main() -> None:
    """rfnry-rag — Retrieval-Augmented Generation + Reasoning-Augmented Classification."""


main.add_command(retrieval_cli, name="retrieval")
main.add_command(reasoning_cli, name="reasoning")


if __name__ == "__main__":
    main()
