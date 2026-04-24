import os

import pytest


def test_rfnry_rag_baml_log_propagates_to_baml_log(monkeypatch):
    from rfnry_rag.common.logging import _propagate_baml_log_env

    monkeypatch.delenv("BAML_LOG", raising=False)
    monkeypatch.setenv("RFNRY_RAG_BAML_LOG", "debug")
    _propagate_baml_log_env()
    assert os.environ["BAML_LOG"] == "debug"


def test_existing_baml_log_is_not_overridden(monkeypatch):
    from rfnry_rag.common.logging import _propagate_baml_log_env

    monkeypatch.setenv("BAML_LOG", "info")
    monkeypatch.setenv("RFNRY_RAG_BAML_LOG", "debug")
    _propagate_baml_log_env()
    assert os.environ["BAML_LOG"] == "info"  # explicit BAML_LOG wins


def test_no_env_no_propagation(monkeypatch):
    from rfnry_rag.common.logging import _propagate_baml_log_env

    monkeypatch.delenv("BAML_LOG", raising=False)
    monkeypatch.delenv("RFNRY_RAG_BAML_LOG", raising=False)
    _propagate_baml_log_env()
    assert "BAML_LOG" not in os.environ


def test_invalid_log_level_raises_configuration_error(monkeypatch) -> None:
    from rfnry_rag.common.errors import ConfigurationError
    from rfnry_rag.common.logging import _resolve_level

    with pytest.raises(ConfigurationError, match="unknown log level"):
        _resolve_level("TRACE")


def test_invalid_baml_log_level_raises_configuration_error(monkeypatch) -> None:
    from rfnry_rag.common.errors import ConfigurationError
    from rfnry_rag.common.logging import _propagate_baml_log_env

    monkeypatch.delenv("BAML_LOG", raising=False)
    monkeypatch.setenv("RFNRY_RAG_BAML_LOG", "verbose")
    with pytest.raises(ConfigurationError, match="RFNRY_RAG_BAML_LOG"):
        _propagate_baml_log_env()


def test_valid_baml_log_level_propagates(monkeypatch) -> None:
    from rfnry_rag.common.logging import _propagate_baml_log_env

    monkeypatch.delenv("BAML_LOG", raising=False)
    monkeypatch.setenv("RFNRY_RAG_BAML_LOG", "WARN")
    _propagate_baml_log_env()
    assert os.environ["BAML_LOG"] == "warn"
