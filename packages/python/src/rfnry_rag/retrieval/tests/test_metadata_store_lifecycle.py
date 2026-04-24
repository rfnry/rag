"""Regression guard: tree indexes must be removed when their source is deleted.

tree_index_json currently lives on the source row, so delete_source already
removes it. This test locks in that behavior — if a future schema change
moves tree_index_json to a separate table, the test will fail unless that
table is wired up with ON DELETE CASCADE (or explicit deletion)."""

from datetime import UTC, datetime

import pytest

from rfnry_rag.retrieval.common.models import Source
from rfnry_rag.retrieval.stores.metadata.sqlalchemy import SQLAlchemyMetadataStore


@pytest.mark.asyncio
async def test_delete_source_also_removes_tree_index(tmp_path) -> None:
    store = SQLAlchemyMetadataStore(f"sqlite:///{tmp_path}/m.db")
    await store.initialize()

    source = Source(
        source_id="s1",
        knowledge_id="k",
        source_type="manual",
        status="completed",
        embedding_model="openai:text-embedding-3-small",
        created_at=datetime.now(UTC),
    )
    await store.create_source(source)

    await store.save_tree_index("s1", '{"nodes": [], "meta": "v1"}')
    assert await store.get_tree_index("s1") == '{"nodes": [], "meta": "v1"}'

    await store.delete_source("s1")

    # The tree index is gone because its storage is coupled to the source row.
    # If this starts failing after a schema change (e.g. tree_index moved to a
    # separate table), wire up ON DELETE CASCADE or delete it explicitly in
    # delete_source — do NOT relax this test.
    assert await store.get_tree_index("s1") is None
    assert await store.get_source("s1") is None


@pytest.mark.asyncio
async def test_delete_source_is_idempotent_on_tree_index(tmp_path) -> None:
    """Deleting a source that never had a tree index is a no-op, not an error."""
    store = SQLAlchemyMetadataStore(f"sqlite:///{tmp_path}/m.db")
    await store.initialize()

    source = Source(
        source_id="s2",
        knowledge_id="k",
        embedding_model="m",
        created_at=datetime.now(UTC),
    )
    await store.create_source(source)

    assert await store.get_tree_index("s2") is None
    await store.delete_source("s2")
    assert await store.get_tree_index("s2") is None


@pytest.mark.asyncio
async def test_find_by_hash_returns_match(tmp_path) -> None:
    store = SQLAlchemyMetadataStore(f"sqlite:///{tmp_path}/m.db")
    await store.initialize()

    source = Source(
        source_id="s3",
        knowledge_id="kb1",
        embedding_model="m",
        file_hash="abc123",
        created_at=datetime.now(UTC),
    )
    await store.create_source(source)

    # Exact match on hash + knowledge_id
    found = await store.find_by_hash("abc123", "kb1")
    assert found is not None
    assert found.source_id == "s3"

    # No match on unknown hash
    assert await store.find_by_hash("nope", "kb1") is None

    # No match when knowledge_id differs
    assert await store.find_by_hash("abc123", "different") is None


async def test_initialization_is_idempotent(tmp_path) -> None:
    """initialize() called twice on the same store must not error."""
    store = SQLAlchemyMetadataStore(url=f"sqlite+aiosqlite:///{tmp_path / 'db.sqlite'}")
    await store.initialize()
    await store.initialize()   # must not raise (would be "duplicate column" today)
    await store.shutdown()


async def test_file_hash_column_has_index(tmp_path) -> None:
    """find_by_hash must hit an index, not a full scan."""
    import sqlalchemy as sa

    store = SQLAlchemyMetadataStore(url=f"sqlite+aiosqlite:///{tmp_path / 'db.sqlite'}")
    await store.initialize()
    async with store._engine.begin() as conn:
        result = await conn.run_sync(
            lambda sync_conn: sa.inspect(sync_conn).get_indexes("rag_sources")
        )
    names = {idx["name"] for idx in result if idx["name"] is not None}
    assert any("file_hash" in n for n in names), f"no index on file_hash, got indexes: {names}"
    await store.shutdown()
