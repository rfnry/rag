"""BOUNDARY_API_KEY handling — first-write-wins, no silent clobbering
across multiple LanguageModelClient instances."""

import os

import pytest

from rfnry_rag.providers.registry import _apply_boundary_api_key


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("BOUNDARY_API_KEY", raising=False)
    yield


def test_apply_boundary_api_key_sets_env_when_unset() -> None:
    _apply_boundary_api_key("key-1")
    assert os.environ["BOUNDARY_API_KEY"] == "key-1"


def test_apply_boundary_api_key_noop_when_none() -> None:
    _apply_boundary_api_key(None)
    assert "BOUNDARY_API_KEY" not in os.environ


def test_apply_boundary_api_key_same_value_is_idempotent() -> None:
    _apply_boundary_api_key("same")
    _apply_boundary_api_key("same")
    assert os.environ["BOUNDARY_API_KEY"] == "same"


def test_apply_boundary_api_key_different_value_raises() -> None:
    """Collision on boundary key now raises so multi-tenant misconfiguration
    can't be silently hidden."""
    from rfnry_rag.exceptions import ConfigurationError

    _apply_boundary_api_key("first")
    with pytest.raises(ConfigurationError, match="boundary_api_key collision"):
        _apply_boundary_api_key("second")
    assert os.environ["BOUNDARY_API_KEY"] == "first"
