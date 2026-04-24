from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click

from rfnry_rag.reasoning.cli.output import get_output_mode

_MAX_DIR_READ_BYTES = 5_000_000

_DIR_READ_EXTENSIONS = (".md", ".txt", ".json")


def _read_directory_as_text(path: Path) -> str:
    """Read all text files in a directory into a single string.

    Raises ValueError if the aggregate size exceeds _MAX_DIR_READ_BYTES to
    prevent runaway memory use when a caller points to a large directory.
    Raises ValueError if any entry resolves outside the directory (symlink
    traversal / path-containment guard).
    """
    path = path.resolve()
    parts: list[str] = []
    total = 0
    for f in sorted(path.iterdir()):
        if not f.is_file():
            continue
        if f.suffix not in _DIR_READ_EXTENSIONS:
            continue
        resolved = f.resolve()
        # Defense in depth: reject symlinks pointing outside the directory.
        if not resolved.is_relative_to(path):
            raise ValueError(f"refusing to read {resolved}: resolved path escapes directory {path}")
        data = f.read_text()
        total += len(data)
        if total > _MAX_DIR_READ_BYTES:
            raise ValueError(f"directory aggregate read exceeds {_MAX_DIR_READ_BYTES} bytes")
        parts.append(f"[{f.name}]\n{data}")
    return "\n\n========\n\n".join(parts)


@click.group()
@click.option("--json", "output_format", flag_value="json", help="Force JSON output.")
@click.option("--pretty", "output_format", flag_value="pretty", help="Force human-readable output.")
@click.option("--config", "config_path", type=click.Path(), default=None, help="Override config file path.")
@click.pass_context
def cli(ctx: click.Context, output_format: str | None, config_path: str | None) -> None:
    """rfnry-rag — CLI for the rfnry-rag SDK."""
    ctx.ensure_object(dict)
    ctx.obj["output_mode"] = get_output_mode(output_format)
    ctx.obj["config_path"] = config_path


def resolve_input(text: str | None, file_path: str | None) -> str:
    """Resolve text input from argument, file, or stdin."""
    if file_path:
        from pathlib import Path

        return Path(file_path).read_text()
    if text:
        return text
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()
    raise click.UsageError("Provide text, --file, or pipe stdin.")


def resolve_references(paths: tuple[str | Path, ...]) -> str:
    """Resolve one or more reference paths (files or folder) into a single string."""
    parts: list[str] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            dir_text = _read_directory_as_text(path)
            if dir_text:
                parts.append(dir_text)
        elif path.is_file():
            parts.append(f"[{path.name}]\n{path.read_text()}")
        else:
            raise click.UsageError(f"Reference not found: {p}")
    if not parts:
        raise click.UsageError("No reference documents found.")
    return "\n\n========\n\n".join(parts)


def run_async(coro):
    """Run an async coroutine from a sync click command."""
    try:
        asyncio.run(coro)
    except KeyboardInterrupt:
        raise SystemExit(130) from None


def main() -> None:
    import rfnry_rag.reasoning.cli.commands.analyze
    import rfnry_rag.reasoning.cli.commands.analyze_context
    import rfnry_rag.reasoning.cli.commands.classify
    import rfnry_rag.reasoning.cli.commands.compliance
    import rfnry_rag.reasoning.cli.commands.evaluate
    import rfnry_rag.reasoning.cli.commands.init
    import rfnry_rag.reasoning.cli.commands.status

    cli()
