import os


def test_rfnry_rag_baml_log_propagates_to_baml_log(monkeypatch):
    from rfnry_rag.common.logging import _propagate_baml_log_env

    monkeypatch.delenv("BAML_LOG", raising=False)
    monkeypatch.setenv("RRAG_BAML_LOG", "debug")
    _propagate_baml_log_env()
    assert os.environ["BAML_LOG"] == "debug"


def test_existing_baml_log_is_not_overridden(monkeypatch):
    from rfnry_rag.common.logging import _propagate_baml_log_env

    monkeypatch.setenv("BAML_LOG", "info")
    monkeypatch.setenv("RRAG_BAML_LOG", "debug")
    _propagate_baml_log_env()
    assert os.environ["BAML_LOG"] == "info"  # explicit BAML_LOG wins


def test_no_env_no_propagation(monkeypatch):
    from rfnry_rag.common.logging import _propagate_baml_log_env

    monkeypatch.delenv("BAML_LOG", raising=False)
    monkeypatch.delenv("RRAG_BAML_LOG", raising=False)
    _propagate_baml_log_env()
    assert "BAML_LOG" not in os.environ
