"""Preset factory tests — these cover the common pipeline shapes so users
don't have to hand-assemble KnowledgeEngineConfig for simple cases."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from rfnry_knowledge.config import KnowledgeEngineConfig
from rfnry_knowledge.ingestion.methods.keyword import KeywordIngestion
from rfnry_knowledge.ingestion.methods.semantic import SemanticIngestion
from rfnry_knowledge.knowledge.engine import KnowledgeEngine
from rfnry_knowledge.retrieval.methods.entity import EntityRetrieval
from rfnry_knowledge.retrieval.methods.keyword import KeywordRetrieval
from rfnry_knowledge.retrieval.methods.semantic import SemanticRetrieval


def _has(methods, cls) -> bool:
    return any(isinstance(m, cls) for m in methods)


def test_vector_only_preset_yields_valid_config() -> None:
    vector_store = MagicMock()
    embeddings = MagicMock()
    embeddings.model = "test-model"

    config = KnowledgeEngine.vector_only(vector_store=vector_store, embeddings=embeddings)

    assert isinstance(config, KnowledgeEngineConfig)
    assert _has(config.ingestion.methods, SemanticIngestion)
    assert _has(config.retrieval.methods, SemanticRetrieval)
    assert not _has(config.ingestion.methods, KeywordIngestion)
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

    assert _has(config.ingestion.methods, KeywordIngestion)
    assert _has(config.retrieval.methods, KeywordRetrieval)
    assert not _has(config.ingestion.methods, SemanticIngestion)


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

    assert _has(config.ingestion.methods, SemanticIngestion)
    assert _has(config.ingestion.methods, KeywordIngestion)
    assert _has(config.retrieval.methods, EntityRetrieval)
    assert config.retrieval.reranker is reranker


def test_hybrid_preset_without_optional_stores() -> None:
    config = KnowledgeEngine.hybrid(
        vector_store=MagicMock(),
        embeddings=MagicMock(model="m"),
    )
    assert _has(config.ingestion.methods, SemanticIngestion)
    assert not _has(config.ingestion.methods, KeywordIngestion)
    assert not _has(config.retrieval.methods, EntityRetrieval)


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
