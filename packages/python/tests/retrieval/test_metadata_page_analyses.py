"""Metadata store: per-page analysis storage via rag_page_analyses table."""

from datetime import UTC, datetime

import pytest

from rfnry_rag.retrieval.common.models import Source
from rfnry_rag.stores.metadata.sqlalchemy import SQLAlchemyMetadataStore


@pytest.fixture
async def store(tmp_path):
    s = SQLAlchemyMetadataStore(url=f"sqlite+aiosqlite:///{tmp_path}/meta.db")
    await s.initialize()
    yield s
    await s.shutdown()


def _fake_source(source_id: str = "s1", knowledge_id: str = "k1") -> Source:
    return Source(
        source_id=source_id,
        knowledge_id=knowledge_id,
        source_type=None,
        status="analyzed",
        embedding_model="fake-384",
        file_hash="deadbeef",
        created_at=datetime.now(UTC),
        source_weight=1.0,
        metadata={"file_type": "pdf", "file_name": "test.pdf"},
    )


@pytest.mark.asyncio
async def test_upsert_and_fetch_page_analyses(store) -> None:
    await store.create_source(_fake_source())

    await store.upsert_page_analyses(
        source_id="s1",
        analyses=[
            {"page_number": 1, "data": {"description": "page 1", "entities": []}},
            {"page_number": 2, "data": {"description": "page 2", "entities": []}},
        ],
    )

    fetched = await store.get_page_analyses(source_id="s1")
    assert len(fetched) == 2
    # Sorted by page_number
    assert fetched[0]["page_number"] == 1
    assert fetched[1]["page_number"] == 2
    assert fetched[0]["data"]["description"] == "page 1"


@pytest.mark.asyncio
async def test_get_single_page_analysis_by_number(store) -> None:
    await store.create_source(_fake_source())
    await store.upsert_page_analyses(
        "s1",
        [
            {"page_number": 5, "data": {"description": "five"}},
        ],
    )
    one = await store.get_page_analysis(source_id="s1", page_number=5)
    assert one is not None
    assert one["description"] == "five"
    missing = await store.get_page_analysis("s1", 99)
    assert missing is None


@pytest.mark.asyncio
async def test_upsert_is_idempotent_on_source_page(store) -> None:
    """Same (source_id, page_number) — second write overwrites first."""
    await store.create_source(_fake_source())
    await store.upsert_page_analyses(
        "s1",
        [
            {"page_number": 1, "data": {"description": "first", "page_hash": "abc"}},
        ],
    )
    await store.upsert_page_analyses(
        "s1",
        [
            {"page_number": 1, "data": {"description": "second", "page_hash": "def"}},
        ],
    )
    fetched = await store.get_page_analysis("s1", 1)
    assert fetched["description"] == "second"
    assert fetched["page_hash"] == "def"


@pytest.mark.asyncio
async def test_get_page_analyses_empty_for_unknown_source(store) -> None:
    rows = await store.get_page_analyses(source_id="nope")
    assert rows == []


@pytest.mark.asyncio
async def test_migration_adds_table_idempotently(tmp_path) -> None:
    """Two initialize() calls on the same DB URL should not error."""
    url = f"sqlite+aiosqlite:///{tmp_path}/meta.db"
    s1 = SQLAlchemyMetadataStore(url=url)
    await s1.initialize()
    await s1.shutdown()
    s2 = SQLAlchemyMetadataStore(url=url)
    await s2.initialize()  # must not raise
    await s2.shutdown()


@pytest.mark.asyncio
async def test_upsert_stores_page_hash_in_indexed_column(store) -> None:
    """page_hash from data is hoisted into the indexed column for fast lookup."""
    await store.create_source(_fake_source())
    await store.upsert_page_analyses(
        "s1",
        [
            {"page_number": 1, "data": {"description": "x", "page_hash": "hash_abc"}},
        ],
    )
    fetched = await store.get_page_analysis("s1", 1)
    # page_hash round-trips through the JSON blob
    assert fetched["page_hash"] == "hash_abc"
