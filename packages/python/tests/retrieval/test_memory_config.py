from types import SimpleNamespace

import pytest

from rfnry_knowledge.config.memory import (
    MemoryEngineConfig,
    MemoryIngestionConfig,
    MemoryRetrievalConfig,
)
from rfnry_knowledge.exceptions import ConfigurationError


def _stub_extractor():
    class _E:
        async def extract(self, *a, **k):
            return ()
    return _E()


def _stub_embeddings():
    return SimpleNamespace(embed=lambda *a, **k: [], embedding_dimension=lambda: 8, model="x")


def _stub_vector_store():
    return object()


def test_ingestion_defaults() -> None:
    c = MemoryIngestionConfig(
        extractor=_stub_extractor(),
        embeddings=_stub_embeddings(),
        vector_store=_stub_vector_store(),
    )
    assert c.dedup_context_top_k == 0
    assert c.semantic_required is True
    assert c.keyword_backend == "bm25"


def test_ingestion_rejects_negative_dedup_top_k() -> None:
    with pytest.raises(ConfigurationError):
        MemoryIngestionConfig(
            extractor=_stub_extractor(),
            embeddings=_stub_embeddings(),
            vector_store=_stub_vector_store(),
            dedup_context_top_k=-1,
        )


def test_retrieval_weights_must_be_non_negative_and_sum_positive() -> None:
    with pytest.raises(ConfigurationError):
        MemoryRetrievalConfig(semantic_weight=0, keyword_weight=0, entity_weight=0)
    with pytest.raises(ConfigurationError):
        MemoryRetrievalConfig(semantic_weight=-1)


def test_ingestion_requires_document_store_when_postgres_fts_keyword() -> None:
    with pytest.raises(ConfigurationError):
        MemoryIngestionConfig(
            extractor=_stub_extractor(),
            embeddings=_stub_embeddings(),
            vector_store=_stub_vector_store(),
            keyword_backend="postgres_fts",
            document_store=None,
        )


def test_ingestion_requires_graph_store_and_entity_provider_when_entity_extraction_set() -> None:
    from rfnry_knowledge.config import EntityIngestionConfig

    with pytest.raises(ConfigurationError):
        MemoryIngestionConfig(
            extractor=_stub_extractor(),
            embeddings=_stub_embeddings(),
            vector_store=_stub_vector_store(),
            entity_extraction=EntityIngestionConfig(),
            graph_store=None,
        )

    with pytest.raises(ConfigurationError):
        MemoryIngestionConfig(
            extractor=_stub_extractor(),
            embeddings=_stub_embeddings(),
            vector_store=_stub_vector_store(),
            entity_extraction=EntityIngestionConfig(),
            graph_store=object(),
            entity_provider=None,
        )


def test_engine_happy_path() -> None:
    ing = MemoryIngestionConfig(
        extractor=_stub_extractor(),
        embeddings=_stub_embeddings(),
        vector_store=_stub_vector_store(),
    )
    cfg = MemoryEngineConfig(
        ingestion=ing,
        retrieval=MemoryRetrievalConfig(),
    )
    assert cfg.ingestion is ing
