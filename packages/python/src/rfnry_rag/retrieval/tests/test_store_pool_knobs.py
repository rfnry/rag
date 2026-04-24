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


# ── T11: reject zero/negative timeouts and pool knobs ────────────────────────


def test_sqlalchemy_metadata_store_rejects_nonpositive_pool_timeout() -> None:
    from rfnry_rag.retrieval.common.errors import ConfigurationError

    with pytest.raises(ConfigurationError, match="pool_timeout"):
        SQLAlchemyMetadataStore(url="sqlite+aiosqlite:///:memory:", pool_timeout=0)

    with pytest.raises(ConfigurationError, match="pool_timeout"):
        SQLAlchemyMetadataStore(url="sqlite+aiosqlite:///:memory:", pool_timeout=-1)


def test_sqlalchemy_metadata_store_rejects_bad_pool_recycle() -> None:
    from rfnry_rag.retrieval.common.errors import ConfigurationError

    with pytest.raises(ConfigurationError, match="pool_recycle"):
        SQLAlchemyMetadataStore(url="sqlite+aiosqlite:///:memory:", pool_recycle=0)

    # -1 is the SQLAlchemy "never recycle" sentinel — must be accepted
    SQLAlchemyMetadataStore(url="sqlite+aiosqlite:///:memory:", pool_recycle=-1)


def test_postgres_document_store_rejects_nonpositive_pool_timeout() -> None:
    from rfnry_rag.retrieval.common.errors import ConfigurationError

    with pytest.raises(ConfigurationError, match="pool_timeout"):
        PostgresDocumentStore(url="sqlite+aiosqlite:///:memory:", pool_timeout=0)

    with pytest.raises(ConfigurationError, match="pool_timeout"):
        PostgresDocumentStore(url="sqlite+aiosqlite:///:memory:", pool_timeout=-5)


def test_postgres_document_store_rejects_bad_pool_recycle() -> None:
    from rfnry_rag.retrieval.common.errors import ConfigurationError

    with pytest.raises(ConfigurationError, match="pool_recycle"):
        PostgresDocumentStore(url="sqlite+aiosqlite:///:memory:", pool_recycle=0)

    # -1 is the SQLAlchemy "never recycle" sentinel — must be accepted
    PostgresDocumentStore(url="sqlite+aiosqlite:///:memory:", pool_recycle=-1)


def test_qdrant_store_rejects_nonpositive_timeout() -> None:
    from rfnry_rag.retrieval.common.errors import ConfigurationError
    from rfnry_rag.retrieval.stores.vector.qdrant import QdrantVectorStore

    with pytest.raises(ConfigurationError, match="timeout"):
        QdrantVectorStore(url="http://fake", api_key="k", timeout=0)


def test_qdrant_store_rejects_nonpositive_scroll_timeout() -> None:
    from rfnry_rag.retrieval.common.errors import ConfigurationError
    from rfnry_rag.retrieval.stores.vector.qdrant import QdrantVectorStore

    with pytest.raises(ConfigurationError, match="scroll_timeout"):
        QdrantVectorStore(url="http://fake", api_key="k", scroll_timeout=0)


def test_qdrant_store_rejects_nonpositive_write_timeout() -> None:
    from rfnry_rag.retrieval.common.errors import ConfigurationError
    from rfnry_rag.retrieval.stores.vector.qdrant import QdrantVectorStore

    with pytest.raises(ConfigurationError, match="write_timeout"):
        QdrantVectorStore(url="http://fake", api_key="k", write_timeout=-1)


def test_neo4j_store_rejects_nonpositive_query_timeout() -> None:
    from rfnry_rag.retrieval.common.errors import ConfigurationError
    from rfnry_rag.retrieval.stores.graph.neo4j import Neo4jGraphStore

    with pytest.raises(ConfigurationError, match="query_timeout"):
        Neo4jGraphStore(uri="bolt://localhost", password="secret", query_timeout=0)


def test_neo4j_store_rejects_nonpositive_connection_timeout() -> None:
    from rfnry_rag.retrieval.common.errors import ConfigurationError
    from rfnry_rag.retrieval.stores.graph.neo4j import Neo4jGraphStore

    with pytest.raises(ConfigurationError, match="connection_timeout"):
        Neo4jGraphStore(uri="bolt://localhost", password="secret", connection_timeout=0.0)


def test_neo4j_store_rejects_nonpositive_connection_acquisition_timeout() -> None:
    from rfnry_rag.retrieval.common.errors import ConfigurationError
    from rfnry_rag.retrieval.stores.graph.neo4j import Neo4jGraphStore

    with pytest.raises(ConfigurationError, match="connection_acquisition_timeout"):
        Neo4jGraphStore(uri="bolt://localhost", password="secret", connection_acquisition_timeout=-1.0)


async def test_qdrant_store_logs_effective_knobs(caplog) -> None:
    import logging
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from rfnry_rag.retrieval.stores.vector.qdrant import QdrantVectorStore

    caplog.set_level(logging.INFO, logger="rfnry_rag.rfnry_rag.retrieval.stores.vector.qdrant")
    store = QdrantVectorStore(
        url="http://fake:6333",
        api_key="secret-key",
        timeout=7,
        scroll_timeout=29,
        write_timeout=31,
        hybrid_prefetch_multiplier=5,
    )
    assert store._client_instance is not None
    store._client_instance.get_collections = AsyncMock(return_value=SimpleNamespace(collections=[]))  # type: ignore[method-assign]
    store._client_instance.create_collection = AsyncMock(return_value=True)  # type: ignore[method-assign]
    await store.initialize(vector_size=1536)
    joined = "\n".join(r.message for r in caplog.records)
    assert "hybrid_prefetch_multiplier=5" in joined
    assert "timeout=7" in joined
    assert "secret-key" not in joined  # api_key must not leak


async def test_neo4j_store_logs_effective_knobs(caplog) -> None:
    import logging
    from unittest.mock import AsyncMock, MagicMock

    from rfnry_rag.retrieval.stores.graph.neo4j import AsyncGraphDatabase, Neo4jGraphStore

    caplog.set_level(logging.INFO, logger="rfnry_rag.rfnry_rag.retrieval.stores.graph.neo4j")
    store = Neo4jGraphStore(
        uri="bolt://localhost:7687",
        password="secret",
        query_timeout=3.0,
        connection_timeout=4.0,
        connection_acquisition_timeout=6.0,
    )

    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.run = AsyncMock()

    mock_driver = MagicMock()
    mock_driver.verify_connectivity = AsyncMock()
    mock_driver.session = MagicMock(return_value=mock_session)

    original_driver_fn = AsyncGraphDatabase.driver

    def mock_driver_fn(*args, **kwargs):
        return mock_driver

    AsyncGraphDatabase.driver = mock_driver_fn  # type: ignore[method-assign]
    try:
        await store.initialize()
    finally:
        AsyncGraphDatabase.driver = original_driver_fn  # type: ignore[method-assign]

    joined = "\n".join(r.message for r in caplog.records)
    assert "query_timeout=3.0" in joined
    assert "connection_timeout=4.0" in joined
    assert "connection_acquisition_timeout=6.0" in joined
    assert "secret" not in joined  # password must not leak
