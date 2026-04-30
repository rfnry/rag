"""Shared fixtures for the retrieval test suite."""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from rfnry_rag.generation.models import QueryResult
from rfnry_rag.server import RagEngine, RagServerConfig

_UNSET: Any = object()


def _default_query_result(answer: str = "an answer") -> QueryResult:
    return QueryResult(answer=answer, sources=[], grounded=True, confidence=0.85)


@pytest.fixture
def make_engine() -> Any:
    """Factory fixture: builds a minimally-wired ``RagEngine`` for unit tests."""

    def _factory(
        *,
        config: Any = _UNSET,
        routing: Any = _UNSET,
        retrieval: Any = _UNSET,
        generation: Any = _UNSET,
        persistence: Any = _UNSET,
        ingestion: Any = _UNSET,
        initialized: bool = True,
        retrieval_service: Any = _UNSET,
        structured_retrieval: Any = _UNSET,
        generation_service: Any = _UNSET,
        step_service: Any = _UNSET,
        knowledge_manager: Any = _UNSET,
        ingestion_service: Any = _UNSET,
        structured_ingestion: Any = _UNSET,
        retrieval_namespace: Any = _UNSET,
        ingestion_namespace: Any = _UNSET,
    ) -> Any:
        if config is _UNSET:
            built: Any = MagicMock(spec=RagServerConfig)
            if retrieval is not _UNSET:
                built.retrieval = retrieval
            if routing is not _UNSET:
                built.routing = routing
            if generation is not _UNSET:
                built.generation = generation
            if persistence is not _UNSET:
                built.persistence = persistence
            if ingestion is not _UNSET:
                built.ingestion = ingestion
            cfg: Any = built
        else:
            cfg = config

        engine = RagEngine.__new__(RagEngine)
        engine._config = cfg
        engine._initialized = initialized

        if retrieval_service is _UNSET:
            rs: Any = AsyncMock()
            cast(Any, rs).retrieve = AsyncMock(return_value=([], None))
            engine._retrieval_service = rs
        else:
            engine._retrieval_service = retrieval_service

        engine._structured_retrieval = None if structured_retrieval is _UNSET else structured_retrieval

        if generation_service is _UNSET:
            gs: Any = AsyncMock()
            cast(Any, gs).generate = AsyncMock(return_value=_default_query_result())
            cast(Any, gs).generate_from_corpus = AsyncMock(return_value=_default_query_result())
            engine._generation_service = gs
        else:
            engine._generation_service = generation_service

        engine._step_service = None if step_service is _UNSET else step_service
        engine._knowledge_manager = None if knowledge_manager is _UNSET else knowledge_manager
        engine._ingestion_service = None if ingestion_service is _UNSET else ingestion_service
        engine._structured_ingestion = None if structured_ingestion is _UNSET else structured_ingestion
        engine._retrieval_namespace = None if retrieval_namespace is _UNSET else retrieval_namespace
        engine._ingestion_namespace = None if ingestion_namespace is _UNSET else ingestion_namespace

        return engine

    return _factory
