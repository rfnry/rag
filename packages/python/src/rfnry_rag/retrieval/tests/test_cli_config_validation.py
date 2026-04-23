"""CLI config.toml validation — unknown top-level keys must be rejected with
a helpful error, not silently ignored."""

import pytest

from rfnry_rag.retrieval.cli.config import _validate_toml_keys
from rfnry_rag.retrieval.cli.constants import ConfigError


def test_validate_toml_keys_accepts_known_keys() -> None:
    _validate_toml_keys({"persistence": {}, "ingestion": {}, "retrieval": {}})


def test_validate_toml_keys_rejects_unknown_top_level() -> None:
    with pytest.raises(ConfigError, match="Unknown top-level key"):
        _validate_toml_keys({"persistence": {}, "retrievl": {}})  # typo


def test_validate_toml_keys_error_lists_allowed_keys() -> None:
    with pytest.raises(ConfigError, match="Allowed keys"):
        _validate_toml_keys({"generate": {}})


def test_validate_toml_keys_empty_dict_is_fine() -> None:
    _validate_toml_keys({})
