"""Shared fixtures for the retrieval test suite.

The seven retrieval-test files that bypass ``RagEngine.initialize()`` to
unit-test internal dispatch (``test_routing_auto_mode``, ``test_routing_direct_mode``,
``test_routing_hybrid_mode``, ``test_iterative_retrieval``, ``test_confidence_expansion``,
``test_raptor_engine_api``, ``test_tree_search_fusion``) all built a thin
``_make_engine`` helper that did the same scaffolding job: allocate
``RagEngine.__new__(RagEngine)``, attach a config (``MagicMock(spec=RagServerConfig)``
or a ``SimpleNamespace`` for raptor / tree-search), wire up the private
service attributes that ``__init__`` would normally leave as ``None``, and
return the engine typed as ``Any`` so tests can poke ``AsyncMock`` assertion
helpers on attributes typed as concrete services.

The ``make_engine`` factory exposed here consolidates that scaffolding into
a single, override-rich callable. Each call site passes only the knobs it
cares about; the factory wires the rest with sensible no-op defaults.

The factory deliberately returns ``Any`` (mirroring every original helper).
``RagEngine`` private attributes are typed as concrete service classes;
returning ``Any`` lets tests assign ``AsyncMock`` to them and call
``assert_awaited_once`` etc. without mypy fighting the engine class shape.
"""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from rfnry_rag.retrieval.modules.generation.models import QueryResult
from rfnry_rag.retrieval.server import RagEngine, RagServerConfig

# Sentinel object used to distinguish "caller did not pass this kwarg" from
# "caller explicitly passed ``None``" (which means: leave the attribute as
# ``None`` on the engine). Plain ``None`` cannot serve both roles because
# several of the original helpers explicitly set service attributes to
# ``None`` (``engine._knowledge_manager = None`` in test_routing_direct_mode).
_UNSET: Any = object()


def _default_query_result(answer: str = "an answer") -> QueryResult:
    return QueryResult(answer=answer, sources=[], grounded=True, confidence=0.85)


@pytest.fixture
def make_engine() -> Any:
    """Factory fixture: builds a minimally-wired ``RagEngine`` for unit tests.

    The factory bypasses ``RagEngine.initialize()`` and attaches whichever
    private services the caller specifies. Anything left at the default
    ``_UNSET`` sentinel gets a no-op ``AsyncMock`` (for ``_retrieval_service``
    / ``_generation_service``) or ``None`` (for everything else, matching
    the original helpers' shape).

    Parameters mirror the union of the seven original helpers' surfaces:

    * ``config`` — supply a fully-formed config object directly. When ``None``
      (default), the factory builds a ``MagicMock(spec=RagServerConfig)`` and
      stamps the per-section overrides (``routing``, ``retrieval``,
      ``generation``, ``persistence``, ``ingestion``) onto it. Pass an
      explicit ``SimpleNamespace(...)`` for the raptor / tree-search shape.
    * ``routing`` / ``retrieval`` / ``generation`` / ``persistence`` /
      ``ingestion`` — per-section attributes attached to the auto-built
      config when ``config`` is ``None``. Ignored when ``config`` is given.
    * ``initialized`` — value of ``engine._initialized``. Defaults to
      ``True``; tests that exercise ``__init__`` path can pass ``False``.
    * ``retrieval_service`` / ``generation_service`` / ``knowledge_manager``
      / ``iterative_service`` / ``tree_search_service`` etc. — explicit
      service overrides. Pass an ``AsyncMock`` / ``MagicMock`` to inject a
      specific spy; pass ``None`` to force the attribute to ``None``; omit
      to get the no-op default (an ``AsyncMock`` with sensible returns for
      retrieval/generation, ``None`` for every other service).
    """

    def _factory(
        *,
        config: Any = _UNSET,
        # Per-section overrides applied when ``config`` is _UNSET.
        routing: Any = _UNSET,
        retrieval: Any = _UNSET,
        generation: Any = _UNSET,
        persistence: Any = _UNSET,
        ingestion: Any = _UNSET,
        initialized: bool = True,
        # Explicit service overrides — _UNSET means "factory default".
        retrieval_service: Any = _UNSET,
        structured_retrieval: Any = _UNSET,
        generation_service: Any = _UNSET,
        step_service: Any = _UNSET,
        knowledge_manager: Any = _UNSET,
        iterative_service: Any = _UNSET,
        ingestion_service: Any = _UNSET,
        structured_ingestion: Any = _UNSET,
        retrieval_namespace: Any = _UNSET,
        ingestion_namespace: Any = _UNSET,
        tree_indexing_service: Any = _UNSET,
        tree_search_service: Any = _UNSET,
        raptor_builder: Any = _UNSET,
        raptor_registry: Any = _UNSET,
        answerability_registry: Any = _UNSET,
    ) -> Any:
        # ------------------------------------------------------------------
        # Config assembly.
        # ------------------------------------------------------------------
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
        engine._config = cfg  # type: ignore[assignment]
        engine._initialized = initialized

        # ------------------------------------------------------------------
        # Service attribute assignment.
        # _UNSET → factory default; explicit None → leave as None.
        # ------------------------------------------------------------------

        # Retrieval service — default is an AsyncMock with a no-op `retrieve`
        # (mirrors the auto / direct / hybrid helpers).
        if retrieval_service is _UNSET:
            rs: Any = AsyncMock()
            cast(Any, rs).retrieve = AsyncMock(return_value=([], None))
            engine._retrieval_service = rs
        else:
            engine._retrieval_service = retrieval_service

        engine._structured_retrieval = None if structured_retrieval is _UNSET else structured_retrieval

        # Generation service — default is an AsyncMock with both `generate`
        # and `generate_from_corpus` returning a default QueryResult.
        if generation_service is _UNSET:
            gs: Any = AsyncMock()
            cast(Any, gs).generate = AsyncMock(return_value=_default_query_result())
            cast(Any, gs).generate_from_corpus = AsyncMock(return_value=_default_query_result())
            engine._generation_service = gs
        else:
            engine._generation_service = generation_service

        engine._step_service = None if step_service is _UNSET else step_service
        engine._knowledge_manager = None if knowledge_manager is _UNSET else knowledge_manager
        engine._iterative_service = None if iterative_service is _UNSET else iterative_service
        engine._ingestion_service = None if ingestion_service is _UNSET else ingestion_service
        engine._structured_ingestion = None if structured_ingestion is _UNSET else structured_ingestion
        engine._retrieval_namespace = None if retrieval_namespace is _UNSET else retrieval_namespace
        engine._ingestion_namespace = None if ingestion_namespace is _UNSET else ingestion_namespace
        engine._tree_indexing_service = None if tree_indexing_service is _UNSET else tree_indexing_service
        engine._tree_search_service = None if tree_search_service is _UNSET else tree_search_service
        engine._raptor_builder = None if raptor_builder is _UNSET else raptor_builder
        engine._raptor_registry = None if raptor_registry is _UNSET else raptor_registry
        if answerability_registry is not _UNSET:
            engine._answerability_registry = answerability_registry

        return engine

    return _factory
