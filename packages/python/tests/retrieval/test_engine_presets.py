"""Preset factory tests — these cover the common pipeline shapes so users
don't have to hand-assemble KnowledgeEngineConfig for simple cases."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from rfnry_knowledge.config import KnowledgeEngineConfig
from rfnry_knowledge.ingestion.methods.document import DocumentIngestion
from rfnry_knowledge.ingestion.methods.vector import VectorIngestion
from rfnry_knowledge.knowledge.engine import KnowledgeEngine
from rfnry_knowledge.retrieval.methods.document import DocumentRetrieval
from rfnry_knowledge.retrieval.methods.graph import GraphRetrieval
from rfnry_knowledge.retrieval.methods.vector import VectorRetrieval


def _has(methods, cls) -> bool:
    return any(isinstance(m, cls) for m in methods)


def test_vector_only_preset_yields_valid_config() -> None:
    vector_store = MagicMock()
    embeddings = MagicMock()
    embeddings.model = "test-model"

    config = KnowledgeEngine.vector_only(vector_store=vector_store, embeddings=embeddings)

    assert isinstance(config, KnowledgeEngineConfig)
    assert _has(config.ingestion.methods, VectorIngestion)
    assert _has(config.retrieval.methods, VectorRetrieval)
    assert not _has(config.ingestion.methods, DocumentIngestion)
    assert config.metadata_store is None


def test_vector_only_with_reranker_and_top_k() -> None:
    reranker = MagicMock()
    config = KnowledgeEngine.vector_only(
        vector_store=MagicMock(),
        embeddings=MagicMock(model="m"),
        top_k=20,
        reranker=reranker,
    )
    assert config.retrieval.top_k == 20
    assert config.retrieval.reranker is reranker


def test_document_only_preset() -> None:
    document_store = MagicMock()
    config = KnowledgeEngine.document_only(document_store=document_store)

    assert _has(config.ingestion.methods, DocumentIngestion)
    assert _has(config.retrieval.methods, DocumentRetrieval)
    assert not _has(config.ingestion.methods, VectorIngestion)


def test_hybrid_preset_wires_all_stores() -> None:
    vector_store = MagicMock()
    embeddings = MagicMock(model="m")
    document_store = MagicMock()
    graph_store = MagicMock()
    reranker = MagicMock()

    config = KnowledgeEngine.hybrid(
        vector_store=vector_store,
        embeddings=embeddings,
        document_store=document_store,
        graph_store=graph_store,
        reranker=reranker,
    )

    assert _has(config.ingestion.methods, VectorIngestion)
    assert _has(config.ingestion.methods, DocumentIngestion)
    assert _has(config.retrieval.methods, GraphRetrieval)
    assert config.retrieval.reranker is reranker


def test_hybrid_preset_without_optional_stores() -> None:
    config = KnowledgeEngine.hybrid(
        vector_store=MagicMock(),
        embeddings=MagicMock(model="m"),
    )
    assert _has(config.ingestion.methods, VectorIngestion)
    assert not _has(config.ingestion.methods, DocumentIngestion)
    assert not _has(config.retrieval.methods, GraphRetrieval)


def test_presets_return_usable_config_for_engine_construction() -> None:
    """Smoke test: the preset configs must satisfy KnowledgeEngine._validate_config."""
    config = KnowledgeEngine.vector_only(vector_store=MagicMock(), embeddings=MagicMock(model="m"))
    KnowledgeEngine(config)._validate_config()

    config2 = KnowledgeEngine.document_only(document_store=MagicMock())
    KnowledgeEngine(config2)._validate_config()


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

    config = KnowledgeEngine.vector_only(vector_store=vector_store, embeddings=embeddings)
    engine = KnowledgeEngine(config)

    await engine.initialize()
    assert engine._initialized
