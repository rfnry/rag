"""Shared CLI utilities for retrieval and reasoning CLIs."""

from __future__ import annotations

import os
import sys
from enum import Enum
from pathlib import Path


class ConfigError(Exception):
    """CLI configuration error."""


CONFIG_DIR = Path.home() / ".config" / "rfnry_rag"
CONFIG_FILE = CONFIG_DIR / "config.toml"
ENV_FILE = CONFIG_DIR / ".env"


class OutputMode(Enum):
    JSON = "json"
    PRETTY = "pretty"


def get_output_mode(explicit: str | None) -> OutputMode:
    """Determine output mode: explicit flag > TTY detection."""
    if explicit == "json":
        return OutputMode.JSON
    if explicit == "pretty":
        return OutputMode.PRETTY
    return OutputMode.PRETTY if sys.stdout.isatty() else OutputMode.JSON


def get_api_key(env_var: str, provider_name: str) -> str:
    """Read API key from env var, raising ConfigError with a CLI-friendly
    message when absent. Used by both retrieval and reasoning CLI loaders."""
    key = os.environ.get(env_var, "")
    if not key:
        raise ConfigError(f"{env_var} not set — required for {provider_name}. Add it to {ENV_FILE}")
    return key


def load_dotenv(path: Path) -> None:
    """Load KEY=VALUE pairs from a .env file into os.environ."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        if key not in os.environ:
            os.environ[key] = value
