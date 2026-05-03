"""User query text logging must be gated behind KNWL_LOG_QUERIES=true."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from rfnry_knowledge.retrieval.search.service import RetrievalService


@pytest.fixture(autouse=True)
def _reset_query_log_env(monkeypatch):
    monkeypatch.delenv("KNWL_LOG_QUERIES", raising=False)
    yield


@pytest.mark.asyncio
async def test_query_text_not_logged_without_opt_in(monkeypatch) -> None:
    captured: list[str] = []

    method = SimpleNamespace(name="m", weight=1.0, top_k=None, search=AsyncMock(return_value=[]))
    service = RetrievalService(retrieval_methods=[method], top_k=5)

    def fake_info(fmt, *args, **kwargs):
        captured.append(fmt % args if args else fmt)

    monkeypatch.setattr(
        "rfnry_knowledge.retrieval.search.service.logger",
        SimpleNamespace(info=fake_info, exception=lambda *a, **k: None, warning=lambda *a, **k: None),
    )

    await service.retrieve(query="secret customer query", knowledge_id="k1")

    joined = "\n".join(captured)
    assert "secret customer query" not in joined
    assert "len=" in joined


@pytest.mark.asyncio
async def test_query_text_logged_when_opted_in(monkeypatch) -> None:
    monkeypatch.setenv("KNWL_LOG_QUERIES", "true")

    captured: list[str] = []
    method = SimpleNamespace(name="m", weight=1.0, top_k=None, search=AsyncMock(return_value=[]))
    service = RetrievalService(retrieval_methods=[method], top_k=5)

    def fake_info(fmt, *args, **kwargs):
        captured.append(fmt % args if args else fmt)

    monkeypatch.setattr(
        "rfnry_knowledge.retrieval.search.service.logger",
        SimpleNamespace(info=fake_info, exception=lambda *a, **k: None, warning=lambda *a, **k: None),
    )

    await service.retrieve(query="secret customer query", knowledge_id="k1")

    joined = "\n".join(captured)
    assert "secret customer query" in joined
