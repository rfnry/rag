"""Preset factory tests — these cover the common pipeline shapes so users
don't have to hand-assemble RagServerConfig for simple cases."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from rfnry_rag.retrieval.server import RagEngine, RagServerConfig


def test_vector_only_preset_yields_valid_config() -> None:
    vector_store = MagicMock()
    embeddings = MagicMock()
    embeddings.model = "test-model"

    config = RagEngine.vector_only(vector_store=vector_store, embeddings=embeddings)

    assert isinstance(config, RagServerConfig)
    assert config.persistence.vector_store is vector_store
    assert config.ingestion.embeddings is embeddings
    assert config.persistence.document_store is None
    assert config.persistence.graph_store is None


def test_vector_only_with_reranker_and_top_k() -> None:
    reranker = MagicMock()
    config = RagEngine.vector_only(
        vector_store=MagicMock(),
        embeddings=MagicMock(model="m"),
        top_k=20,
        reranker=reranker,
    )
    assert config.retrieval.top_k == 20
    assert config.retrieval.reranker is reranker


def test_document_only_preset() -> None:
    document_store = MagicMock()
    config = RagEngine.document_only(document_store=document_store)

    assert config.persistence.document_store is document_store
    assert config.persistence.vector_store is None
    assert config.ingestion.embeddings is None


def test_hybrid_preset_wires_all_stores() -> None:
    vector_store = MagicMock()
    embeddings = MagicMock(model="m")
    document_store = MagicMock()
    graph_store = MagicMock()
    reranker = MagicMock()

    config = RagEngine.hybrid(
        vector_store=vector_store,
        embeddings=embeddings,
        document_store=document_store,
        graph_store=graph_store,
        reranker=reranker,
    )

    assert config.persistence.vector_store is vector_store
    assert config.persistence.document_store is document_store
    assert config.persistence.graph_store is graph_store
    assert config.retrieval.reranker is reranker


def test_hybrid_preset_without_optional_stores() -> None:
    config = RagEngine.hybrid(
        vector_store=MagicMock(),
        embeddings=MagicMock(model="m"),
    )
    assert config.persistence.vector_store is not None
    assert config.persistence.document_store is None
    assert config.persistence.graph_store is None


def test_presets_return_usable_config_for_engine_construction() -> None:
    """Smoke test: the preset configs must satisfy RagEngine._validate_config."""
    config = RagEngine.vector_only(vector_store=MagicMock(), embeddings=MagicMock(model="m"))
    # _validate_config is called in RagEngine.initialize; exercising it ensures
    # the preset satisfies cross-config constraints.
    RagEngine(config)._validate_config()

    config2 = RagEngine.document_only(document_store=MagicMock())
    RagEngine(config2)._validate_config()


@pytest.mark.asyncio
async def test_vector_only_preset_initializes_without_generation() -> None:
    """Regression: default grounding_threshold=0.5 must not require an LM client
    when generation is not enabled — retrieval-only presets must initialize cleanly."""
    vector_store = MagicMock()
    vector_store.initialize = AsyncMock()
    vector_store.collections = ["knowledge"]
    embeddings = MagicMock()
    embeddings.model = "test"
    embeddings.embedding_dimension = AsyncMock(return_value=1536)

    config = RagEngine.vector_only(vector_store=vector_store, embeddings=embeddings)
    engine = RagEngine(config)

    await engine.initialize()
    assert engine._initialized
