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
