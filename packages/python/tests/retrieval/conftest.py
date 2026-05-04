"""Shared fixtures for the retrieval test suite."""

from __future__ import annotations

from typing import Any, cast
from typing import Any as _Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from rfnry_knowledge.config import KnowledgeEngineConfig
from rfnry_knowledge.generation.models import QueryResult
from rfnry_knowledge.knowledge.engine import KnowledgeEngine
from rfnry_knowledge.observability import NullSink as _ObsNullSink
from rfnry_knowledge.observability import Observability
from rfnry_knowledge.telemetry import NullTelemetrySink as _TelNullSink
from rfnry_knowledge.telemetry import Telemetry

_UNSET: Any = object()


def _default_query_result(answer: str = "an answer") -> QueryResult:
    return QueryResult(answer=answer, sources=[], grounded=True, confidence=0.85)


@pytest.fixture
def make_engine() -> Any:
    """Factory fixture: builds a minimally-wired ``KnowledgeEngine`` for unit tests."""

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
        knowledge_manager: Any = _UNSET,
        ingestion_service: Any = _UNSET,
        structured_ingestion: Any = _UNSET,
        retrieval_namespace: Any = _UNSET,
        ingestion_namespace: Any = _UNSET,
    ) -> Any:
        if config is _UNSET:
            built: Any = MagicMock(spec=KnowledgeEngineConfig)
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

        engine = KnowledgeEngine.__new__(KnowledgeEngine)
        engine._config = cfg
        engine._observability = Observability(sink=_ObsNullSink())
        engine._telemetry = Telemetry(sink=_TelNullSink())
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

        engine._knowledge_manager = None if knowledge_manager is _UNSET else knowledge_manager
        engine._ingestion_service = None if ingestion_service is _UNSET else ingestion_service
        engine._structured_ingestion = None if structured_ingestion is _UNSET else structured_ingestion
        engine._retrieval_namespace = None if retrieval_namespace is _UNSET else retrieval_namespace
        engine._ingestion_namespace = None if ingestion_namespace is _UNSET else ingestion_namespace

        return engine

    return _factory


class _FakeMemoryVectorStore:
    def __init__(self) -> None:
        self.points: list[_Any] = []
        self.deleted: list[dict] = []
        self._scroll_results: list[_Any] = []
        self._search_results: list[_Any] = []

    async def initialize(self, vector_size: int) -> None: ...
    async def upsert(self, points): self.points.extend(points)
    async def delete(self, filters):
        self.deleted.append(filters)
        return 1
    async def search(self, vector, top_k=10, filters=None): return list(self._search_results)
    async def hybrid_search(self, *a, **k): return []
    async def retrieve(self, point_ids): return []
    async def scroll(self, filters=None, limit=100, offset=None):
        return list(self._scroll_results), None
    async def count(self, filters=None): return len(self.points)
    async def set_payload(self, *a, **k): ...
    async def shutdown(self): ...


class _FakeMemoryEmbeddings:
    model = "fake"
    name = "fake:fake"
    async def embed(self, texts):
        from rfnry_knowledge.providers import EmbeddingResult
        return EmbeddingResult(vectors=[[0.1] * 8 for _ in texts], usage=None)
    async def embedding_dimension(self): return 8


class _StubMemoryExtractor:
    def __init__(self, items=()) -> None:
        self.items = list(items)
        self.calls: list[_Any] = []
    async def extract(self, interaction, existing_memories=()):
        self.calls.append((interaction, existing_memories))
        return tuple(self.items)


@pytest.fixture
def fake_memory_vector_store():
    return _FakeMemoryVectorStore()


@pytest.fixture
def fake_memory_embeddings():
    return _FakeMemoryEmbeddings()


@pytest.fixture
def stub_memory_extractor_factory():
    def _make(items=()):
        return _StubMemoryExtractor(items)
    return _make


@pytest.fixture
def memory_cfg_factory(fake_memory_vector_store, fake_memory_embeddings):
    from rfnry_knowledge.config.memory import (
        MemoryEngineConfig,
        MemoryIngestionConfig,
        MemoryRetrievalConfig,
    )

    def _make(extractor=None, vector_store=None):
        extractor = extractor or _StubMemoryExtractor()
        return MemoryEngineConfig(
            ingestion=MemoryIngestionConfig(
                extractor=extractor,
                embeddings=fake_memory_embeddings,
                vector_store=vector_store or fake_memory_vector_store,
            ),
            retrieval=MemoryRetrievalConfig(),
        )
    return _make
