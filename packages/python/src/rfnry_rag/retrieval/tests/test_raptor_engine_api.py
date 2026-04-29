"""R2.2 polish — RagEngine.build_raptor_index API guards.

R2.2 lands the engine-level ``build_raptor_index`` API that lazily constructs
``RaptorTreeBuilder`` + ``RaptorTreeRegistry`` on first call. The runtime
checks at the engine boundary (config + dependency presence) belong to the
public surface and need their own coverage so a future regression renaming
``ConfigurationError`` raise sites doesn't slip past unnoticed.

Bias-term hygiene: fixtures use neutral identifiers (``kb-1``, ``topic_a``).
No domain-specific vocabulary anywhere.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from rfnry_rag.retrieval.common.errors import ConfigurationError
from rfnry_rag.retrieval.common.language_model import (
    LanguageModelClient,
    LanguageModelProvider,
)
from rfnry_rag.retrieval.modules.ingestion.methods.raptor.config import RaptorConfig
from rfnry_rag.retrieval.server import RagEngine


def _lm_client() -> LanguageModelClient:
    return LanguageModelClient(
        provider=LanguageModelProvider(provider="openai", model="gpt-test", api_key="x"),
    )


def _make_engine(
    *,
    raptor: RaptorConfig | None = None,
    vector_store: Any = "stub",
    embeddings: Any = "stub",
    metadata_store: Any = "stub_sqlalchemy",
) -> Any:
    """Build a minimally-wired RagEngine bypassing initialize().

    The default sentinel values keep all four engine-API guards happy when
    not under test; individual tests override the field they're exercising.
    Returns ``Any`` so tests can poke private attributes without typecheck
    fighting the engine class shape (mirrors ``test_routing_direct_mode.py``).
    """
    cfg = raptor if raptor is not None else RaptorConfig()
    ingestion = SimpleNamespace(
        raptor=cfg,
        embeddings=embeddings if embeddings != "stub" else MagicMock(),
    )
    persistence = SimpleNamespace(
        vector_store=(
            None if vector_store is None else (vector_store if vector_store != "stub" else MagicMock())
        ),
        metadata_store=(
            None
            if metadata_store is None
            else (metadata_store if metadata_store != "stub_sqlalchemy" else _make_sqlalchemy_store())
        ),
    )
    config = SimpleNamespace(persistence=persistence, ingestion=ingestion)

    engine = RagEngine.__new__(RagEngine)
    engine._config = config  # type: ignore[assignment]
    engine._initialized = True
    engine._knowledge_manager = MagicMock()
    engine._raptor_builder = None
    engine._raptor_registry = None
    return engine


def _make_sqlalchemy_store() -> Any:
    """Return an object that ``isinstance(_, SQLAlchemyMetadataStore)`` accepts."""
    from rfnry_rag.retrieval.stores.metadata.sqlalchemy import SQLAlchemyMetadataStore

    return SQLAlchemyMetadataStore.__new__(SQLAlchemyMetadataStore)


# ---------------------------------------------------------------------------
# Test 1: enabled=False raises ConfigurationError mentioning "enabled".
# ---------------------------------------------------------------------------


async def test_build_raptor_index_raises_when_disabled() -> None:
    """RaptorConfig.enabled=False at API boundary -> ConfigurationError."""
    engine = _make_engine(raptor=RaptorConfig(enabled=False))
    with pytest.raises(ConfigurationError, match="enabled"):
        await engine.build_raptor_index("kb-1")


# ---------------------------------------------------------------------------
# Test 2: enabled=True with summary_model nulled post-init -> ConfigurationError.
# ---------------------------------------------------------------------------


async def test_build_raptor_index_raises_when_summary_model_missing_at_runtime() -> None:
    """Defensive: a consumer mutated summary_model to None after construction."""
    cfg = RaptorConfig(enabled=True, summary_model=_lm_client())
    cfg.summary_model = None  # post-init mutation bypasses dataclass validation
    engine = _make_engine(raptor=cfg)
    with pytest.raises(ConfigurationError, match="summary_model"):
        await engine.build_raptor_index("kb-1")


# ---------------------------------------------------------------------------
# Test 3: vector_store=None -> ConfigurationError mentioning "vector_store".
# ---------------------------------------------------------------------------


async def test_build_raptor_index_raises_when_vector_store_unavailable() -> None:
    cfg = RaptorConfig(enabled=True, summary_model=_lm_client())
    engine = _make_engine(raptor=cfg, vector_store=None)
    with pytest.raises(ConfigurationError, match="vector_store"):
        await engine.build_raptor_index("kb-1")


# ---------------------------------------------------------------------------
# Test 4: metadata_store not SQLAlchemyMetadataStore -> ConfigurationError.
# ---------------------------------------------------------------------------


async def test_build_raptor_index_raises_when_metadata_store_not_sqlalchemy() -> None:
    """RaptorTreeRegistry needs the SQLAlchemy schema; reject other impls."""

    class _OtherMetadataStore:
        pass

    cfg = RaptorConfig(enabled=True, summary_model=_lm_client())
    engine = _make_engine(raptor=cfg, metadata_store=_OtherMetadataStore())
    with pytest.raises(ConfigurationError, match="SQLAlchemyMetadataStore"):
        await engine.build_raptor_index("kb-1")


# ---------------------------------------------------------------------------
# Test 5: enabled=False at __init__ -> _raptor_builder / _raptor_registry None.
# ---------------------------------------------------------------------------


def test_engine_init_does_not_construct_raptor_when_disabled() -> None:
    """Default-off path: ``__init__`` does not allocate RAPTOR collaborators."""
    from rfnry_rag.retrieval.server import (
        IngestionConfig,
        PersistenceConfig,
        RagServerConfig,
        RetrievalConfig,
    )

    config = RagServerConfig(
        persistence=PersistenceConfig(),
        ingestion=IngestionConfig(),
        retrieval=RetrievalConfig(),
    )
    engine = RagEngine(config)
    # Default RaptorConfig has enabled=False; no builder / registry should
    # exist until the consumer calls ``build_raptor_index``.
    assert engine._raptor_builder is None  # type: ignore[attr-defined]
    assert engine._raptor_registry is None  # type: ignore[attr-defined]
