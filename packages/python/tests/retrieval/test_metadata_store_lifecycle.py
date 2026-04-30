"""Regression guard: tree indexes must be removed when their source is deleted.

tree_index_json currently lives on the source row, so delete_source already
removes it. This test locks in that behavior — if a future schema change
moves tree_index_json to a separate table, the test will fail unless that
table is wired up with ON DELETE CASCADE (or explicit deletion)."""

from datetime import UTC, datetime
from typing import Any

import pytest

from rfnry_rag.retrieval.common.models import Source
from rfnry_rag.stores.metadata.sqlalchemy import SQLAlchemyMetadataStore


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
    await store.initialize()  # must not raise (would be "duplicate column" today)
    await store.shutdown()


async def test_get_tree_indexes_returns_mapping(tmp_path) -> None:
    store = SQLAlchemyMetadataStore(url=f"sqlite+aiosqlite:///{tmp_path / 'db.sqlite'}")
    await store.initialize()

    # Seed two sources with tree indexes; one without.
    base: dict[str, Any] = dict(knowledge_id="kb1", embedding_model="m", created_at=datetime.now(UTC))
    await store.create_source(Source(source_id="s1", **base))
    await store.create_source(Source(source_id="s2", **base))
    await store.create_source(Source(source_id="s3", **base))

    await store.save_tree_index("s1", '{"nodes":[],"meta":"s1"}')
    await store.save_tree_index("s2", '{"nodes":[],"meta":"s2"}')
    # s3 intentionally has no tree index

    result = await store.get_tree_indexes(["s1", "s2", "s3", "missing"])
    assert result["s1"] == '{"nodes":[],"meta":"s1"}'
    assert result["s2"] == '{"nodes":[],"meta":"s2"}'
    assert result["s3"] is None
    assert result["missing"] is None

    # Empty input must return empty dict without hitting the DB.
    assert await store.get_tree_indexes([]) == {}

    await store.shutdown()


async def test_list_source_ids_returns_only_ids(tmp_path) -> None:
    store = SQLAlchemyMetadataStore(url=f"sqlite+aiosqlite:///{tmp_path / 'db.sqlite'}")
    await store.initialize()

    # Seed 2 sources in kb1, 1 source in kb2.
    now = datetime.now(UTC)
    await store.create_source(Source(source_id="s1", knowledge_id="kb1", embedding_model="m", created_at=now))
    await store.create_source(Source(source_id="s2", knowledge_id="kb1", embedding_model="m", created_at=now))
    await store.create_source(Source(source_id="s3", knowledge_id="kb2", embedding_model="m", created_at=now))

    ids_kb1 = await store.list_source_ids(knowledge_id="kb1")
    assert set(ids_kb1) == {"s1", "s2"}

    ids_all = await store.list_source_ids()
    assert len(ids_all) == 3

    ids_kb_unknown = await store.list_source_ids(knowledge_id="unknown")
    assert ids_kb_unknown == []

    await store.shutdown()


async def test_file_hash_column_has_index(tmp_path) -> None:
    """find_by_hash must hit an index, not a full scan."""
    import sqlalchemy as sa

    store = SQLAlchemyMetadataStore(url=f"sqlite+aiosqlite:///{tmp_path / 'db.sqlite'}")
    await store.initialize()
    async with store._engine.begin() as conn:
        result = await conn.run_sync(lambda sync_conn: sa.inspect(sync_conn).get_indexes("rag_sources"))
    names = {idx["name"] for idx in result if idx["name"] is not None}
    assert any("file_hash" in n for n in names), f"no index on file_hash, got indexes: {names}"
    await store.shutdown()
