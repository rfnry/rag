"""BOUNDARY_API_KEY handling — first-write-wins, no silent clobbering
across multiple LanguageModelClient instances."""

import logging
import os

import pytest

from rfnry_rag.common.language_model import _apply_boundary_api_key


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


def test_apply_boundary_api_key_different_value_warns_and_preserves_first(
    caplog,
) -> None:
    _apply_boundary_api_key("first")
    with caplog.at_level(logging.WARNING, logger="rfnry_rag.common.language_model"):
        _apply_boundary_api_key("second")
    assert os.environ["BOUNDARY_API_KEY"] == "first"
    assert any("boundary_api_key already set" in rec.message for rec in caplog.records)
