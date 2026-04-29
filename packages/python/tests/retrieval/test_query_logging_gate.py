"""User query text logging must be gated behind RFNRY_RAG_LOG_QUERIES=true."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from rfnry_rag.retrieval.modules.retrieval.search.service import RetrievalService


@pytest.fixture(autouse=True)
def _reset_query_log_env(monkeypatch):
    monkeypatch.delenv("RFNRY_RAG_LOG_QUERIES", raising=False)
    yield


@pytest.mark.asyncio
async def test_query_text_not_logged_without_opt_in(monkeypatch) -> None:
    captured: list[str] = []

    method = SimpleNamespace(name="m", weight=1.0, top_k=None, search=AsyncMock(return_value=[]))
    service = RetrievalService(retrieval_methods=[method], top_k=5)

    def fake_info(fmt, *args, **kwargs):
        captured.append(fmt % args if args else fmt)

    monkeypatch.setattr(
        "rfnry_rag.retrieval.modules.retrieval.search.service.logger",
        SimpleNamespace(info=fake_info, exception=lambda *a, **k: None, warning=lambda *a, **k: None),
    )

    await service.retrieve(query="secret customer query", knowledge_id="k1")

    joined = "\n".join(captured)
    assert "secret customer query" not in joined
    assert "len=" in joined  # length is still logged


@pytest.mark.asyncio
async def test_query_text_logged_when_opted_in(monkeypatch) -> None:
    monkeypatch.setenv("RFNRY_RAG_LOG_QUERIES", "true")

    captured: list[str] = []
    method = SimpleNamespace(name="m", weight=1.0, top_k=None, search=AsyncMock(return_value=[]))
    service = RetrievalService(retrieval_methods=[method], top_k=5)

    def fake_info(fmt, *args, **kwargs):
        captured.append(fmt % args if args else fmt)

    monkeypatch.setattr(
        "rfnry_rag.retrieval.modules.retrieval.search.service.logger",
        SimpleNamespace(info=fake_info, exception=lambda *a, **k: None, warning=lambda *a, **k: None),
    )

    await service.retrieve(query="secret customer query", knowledge_id="k1")

    joined = "\n".join(captured)
    assert "secret customer query" in joined


@pytest.mark.asyncio
async def test_step_back_does_not_log_query_without_opt_in(monkeypatch) -> None:
    """step_back.rewrite must honor RFNRY_RAG_LOG_QUERIES like the service does."""
    from rfnry_rag.retrieval.modules.retrieval.search.rewriting import step_back as step_back_mod
    from rfnry_rag.retrieval.modules.retrieval.search.rewriting.step_back import StepBackRewriting

    captured: list[str] = []

    def fake_info(fmt, *args, **kwargs):
        captured.append(fmt % args if args else fmt)

    monkeypatch.setattr(
        step_back_mod,
        "logger",
        SimpleNamespace(info=fake_info, exception=lambda *a, **k: None, warning=lambda *a, **k: None),
    )

    async def fake_generate(_prompt, baml_options):
        return SimpleNamespace(broader_query="BROADER-VERSION")

    monkeypatch.setattr(step_back_mod.b, "GenerateStepBackQuery", fake_generate)
    monkeypatch.setattr(step_back_mod, "build_registry", lambda _lm: None)

    lm = SimpleNamespace()
    rewriter = StepBackRewriting(lm_client=lm)  # type: ignore[arg-type]
    await rewriter.rewrite(query="secret customer query about refund")

    joined = "\n".join(captured)
    assert "secret customer query" not in joined
    assert "BROADER-VERSION" not in joined
    assert "orig_len=" in joined  # length still logged


@pytest.mark.asyncio
async def test_step_back_logs_query_when_opted_in(monkeypatch) -> None:
    monkeypatch.setenv("RFNRY_RAG_LOG_QUERIES", "true")

    from rfnry_rag.retrieval.modules.retrieval.search.rewriting import step_back as step_back_mod
    from rfnry_rag.retrieval.modules.retrieval.search.rewriting.step_back import StepBackRewriting

    captured: list[str] = []

    def fake_info(fmt, *args, **kwargs):
        captured.append(fmt % args if args else fmt)

    monkeypatch.setattr(
        step_back_mod,
        "logger",
        SimpleNamespace(info=fake_info, exception=lambda *a, **k: None, warning=lambda *a, **k: None),
    )

    async def fake_generate(_prompt, baml_options):
        return SimpleNamespace(broader_query="BROADER-VERSION")

    monkeypatch.setattr(step_back_mod.b, "GenerateStepBackQuery", fake_generate)
    monkeypatch.setattr(step_back_mod, "build_registry", lambda _lm: None)

    lm = SimpleNamespace()
    rewriter = StepBackRewriting(lm_client=lm)  # type: ignore[arg-type]
    await rewriter.rewrite(query="secret customer query about refund")

    joined = "\n".join(captured)
    assert "secret customer query" in joined
