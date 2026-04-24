"""Pool-tuning smoke tests — verify the SQL stores accept pool params without
erroring. Pool semantics are only active on PostgreSQL; SQLite uses StaticPool
and silently ignores pool_size/max_overflow."""

import pytest

from rfnry_rag.retrieval.stores.document.postgres import PostgresDocumentStore
from rfnry_rag.retrieval.stores.metadata.sqlalchemy import SQLAlchemyMetadataStore


def test_sqlalchemy_store_accepts_pool_knobs(tmp_path) -> None:
    db = tmp_path / "m.db"
    store = SQLAlchemyMetadataStore(
        f"sqlite:///{db}",
        pool_recycle=900,
        pool_pre_ping=True,
    )
    assert store._engine is not None


def test_postgres_store_accepts_pool_knobs(tmp_path) -> None:
    db = tmp_path / "d.db"
    store = PostgresDocumentStore(
        f"sqlite:///{db}",
        pool_recycle=900,
        pool_pre_ping=True,
    )
    assert store._engine is not None


@pytest.mark.parametrize(
    "store_cls",
    [SQLAlchemyMetadataStore, PostgresDocumentStore],
)
def test_pool_size_applied_on_postgres_url(store_cls) -> None:
    """With a postgresql URL, pool_size and max_overflow are forwarded to the engine."""
    store = store_cls(
        "postgresql://u:p@localhost:5432/db",
        pool_size=20,
        max_overflow=30,
        pool_recycle=600,
    )
    engine = store._engine
    # Sync engine's pool exposes size/overflow via _pool (SQLAlchemy private API).
    # Prefer public pool.size() when available.
    sync_engine = engine.sync_engine
    pool = sync_engine.pool
    assert pool.size() == 20
    # max_overflow isn't exposed publicly; rely on kwargs getting through (not raising).
    # If SQLAlchemy rejected the kwargs, construction above would have failed.


def test_qdrant_store_exposes_per_op_timeouts() -> None:
    from rfnry_rag.retrieval.stores.vector.qdrant import QdrantVectorStore

    store = QdrantVectorStore(
        url="http://localhost:6333",
        api_key="k",
        timeout=5,
        scroll_timeout=45,
        write_timeout=50,
        max_scroll_limit=5_000,
    )
    assert store._timeout == 5
    assert store._scroll_timeout == 45
    assert store._write_timeout == 50
    assert store._max_scroll_limit == 5_000


def test_qdrant_store_warns_on_plaintext_no_auth(caplog) -> None:
    import logging

    from rfnry_rag.retrieval.stores.vector.qdrant import QdrantVectorStore

    caplog.set_level(logging.WARNING, logger="rfnry_rag.rfnry_rag.retrieval.stores.vector.qdrant")

    QdrantVectorStore(url="http://localhost:6333", api_key=None)

    joined = "\n".join(r.message for r in caplog.records)
    assert "plaintext" in joined.lower() or "production" in joined.lower()


def test_qdrant_store_does_not_warn_with_api_key(caplog) -> None:
    import logging

    from rfnry_rag.retrieval.stores.vector.qdrant import QdrantVectorStore

    caplog.set_level(logging.WARNING)
    QdrantVectorStore(url="http://localhost:6333", api_key="k")

    joined = "\n".join(r.message for r in caplog.records)
    assert "plaintext" not in joined.lower()


def test_qdrant_store_does_not_warn_with_https(caplog) -> None:
    import logging

    from rfnry_rag.retrieval.stores.vector.qdrant import QdrantVectorStore

    caplog.set_level(logging.WARNING)
    QdrantVectorStore(url="https://qdrant.example:6333", api_key=None)

    joined = "\n".join(r.message for r in caplog.records)
    assert "plaintext" not in joined.lower()


async def test_metadata_store_logs_effective_pool_knobs(caplog) -> None:
    import logging

    caplog.set_level(logging.INFO, logger="rfnry_rag.rfnry_rag.retrieval.stores.metadata.sqlalchemy")
    store = SQLAlchemyMetadataStore(url="sqlite+aiosqlite:///:memory:", pool_size=7, max_overflow=14)
    await store.initialize()
    await store.shutdown()
    msg = "\n".join(r.message for r in caplog.records)
    assert "pool" in msg.lower()


async def test_document_store_logs_effective_pool_knobs(caplog) -> None:
    import logging

    caplog.set_level(logging.INFO, logger="rfnry_rag.rfnry_rag.retrieval.stores.document.postgres")
    store = PostgresDocumentStore(url="sqlite+aiosqlite:///:memory:", pool_size=5, max_overflow=10)
    await store.initialize()
    await store.shutdown()
    msg = "\n".join(r.message for r in caplog.records)
    assert "pool" in msg.lower()


def test_build_registry_logs_lm_policy(caplog) -> None:
    import logging

    from rfnry_rag.common.language_model import LanguageModelClient, LanguageModelProvider, build_registry

    caplog.set_level(logging.INFO, logger="rfnry_rag.common.language_model")
    client = LanguageModelClient(
        provider=LanguageModelProvider(provider="openai", model="gpt-4o", api_key="sk-test"),
        max_retries=2,
        timeout_seconds=30,
    )
    build_registry(client)
    msg = "\n".join(r.message for r in caplog.records)
    assert "strategy" in msg.lower() or "max_retries" in msg.lower() or "timeout" in msg.lower()


def test_qdrant_hybrid_prefetch_multiplier_defaults_and_validates() -> None:
    from rfnry_rag.retrieval.common.errors import ConfigurationError
    from rfnry_rag.retrieval.stores.vector.qdrant import QdrantVectorStore

    default = QdrantVectorStore(url="http://fake", api_key="k")
    assert default._hybrid_prefetch_multiplier == 4

    custom = QdrantVectorStore(url="http://fake", api_key="k", hybrid_prefetch_multiplier=8)
    assert custom._hybrid_prefetch_multiplier == 8

    with pytest.raises(ConfigurationError, match="hybrid_prefetch_multiplier"):
        QdrantVectorStore(url="http://fake", api_key="k", hybrid_prefetch_multiplier=0)
