from __future__ import annotations

import json

import click

from rfnry_rag.retrieval.cli import cli
from rfnry_rag.retrieval.cli.constants import CONFIG_DIR

SESSIONS_DIR = CONFIG_DIR / "sessions"


def load_session(name: str) -> list[tuple[str, str]]:
    path = SESSIONS_DIR / f"{name}.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    return [(turn["query"], turn["answer"]) for turn in data]


def save_turn(name: str, query: str, answer: str) -> None:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = SESSIONS_DIR / f"{name}.json"
    data = json.loads(path.read_text()) if path.exists() else []
    data.append({"query": query, "answer": answer})
    path.write_text(json.dumps(data, indent=2))


def list_sessions() -> list[str]:
    if not SESSIONS_DIR.exists():
        return []
    return sorted(p.stem for p in SESSIONS_DIR.glob("*.json"))


def clear_session(name: str) -> bool:
    path = SESSIONS_DIR / f"{name}.json"
    if path.exists():
        path.unlink()
        return True
    return False


@cli.group()
def session() -> None:
    """Manage query sessions."""


@session.command("list")
def list_cmd() -> None:
    """List active sessions."""
    sessions = list_sessions()
    if not sessions:
        click.echo("No active sessions.")
        return
    for name in sessions:
        path = SESSIONS_DIR / f"{name}.json"
        data = json.loads(path.read_text())
        click.echo(f"  {name} ({len(data)} turns)")


@session.command("clear")
@click.argument("name")
def clear_cmd(name: str) -> None:
    """Clear a session's history."""
    if clear_session(name):
        click.echo(f"Session '{name}' cleared.")
    else:
        click.echo(f"Session '{name}' not found.")
