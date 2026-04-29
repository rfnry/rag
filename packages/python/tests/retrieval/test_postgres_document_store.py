import inspect

import pytest

from rfnry_rag.retrieval.stores.document.postgres import PostgresDocumentStore


@pytest.fixture
async def store(tmp_path):
    db_path = tmp_path / "test_docs.db"
    s = PostgresDocumentStore(f"sqlite:///{db_path}")
    await s.initialize()
    yield s
    await s.shutdown()


async def test_store_and_search(store):
    await store.store_content(
        source_id="src-001",
        knowledge_id="kb-1",
        source_type="manuals",
        title="Pump Model X Manual",
        content="The FBD-20254-MERV13 filter has a pressure drop of 0.25 inches WG at 500 FPM.",
    )
    results = await store.search_content(query="FBD-20254", knowledge_id="kb-1", top_k=5)
    assert len(results) >= 1
    assert results[0].source_id == "src-001"
    assert "FBD-20254" in results[0].excerpt


async def test_search_empty(store):
    results = await store.search_content(query="nonexistent", knowledge_id="kb-1")
    assert results == []


async def test_search_scoped_by_knowledge_id(store):
    await store.store_content(
        source_id="src-a",
        knowledge_id="kb-1",
        source_type="manuals",
        title="Doc A",
        content="Filter model ABC-123 specs here.",
    )
    await store.store_content(
        source_id="src-b",
        knowledge_id="kb-2",
        source_type="manuals",
        title="Doc B",
        content="Filter model ABC-123 also referenced here.",
    )
    results = await store.search_content(query="ABC-123", knowledge_id="kb-1")
    assert len(results) == 1
    assert results[0].source_id == "src-a"


async def test_search_scoped_by_source_type(store):
    await store.store_content(
        source_id="src-m",
        knowledge_id="kb-1",
        source_type="manuals",
        title="Manual",
        content="Part number XYZ-789 in manual.",
    )
    await store.store_content(
        source_id="src-d",
        knowledge_id="kb-1",
        source_type="drawings",
        title="Drawing",
        content="Part number XYZ-789 in drawing.",
    )
    results = await store.search_content(query="XYZ-789", knowledge_id="kb-1", source_type="manuals")
    assert len(results) == 1
    assert results[0].source_id == "src-m"


async def test_delete_content(store):
    await store.store_content(
        source_id="src-del",
        knowledge_id="kb-1",
        source_type="manuals",
        title="To Delete",
        content="Unique searchable content for deletion test.",
    )
    results = await store.search_content(query="deletion test", knowledge_id="kb-1")
    assert len(results) == 1
    await store.delete_content("src-del")
    results = await store.search_content(query="deletion test", knowledge_id="kb-1")
    assert results == []


async def test_store_content_upsert(store):
    await store.store_content(
        source_id="src-up",
        knowledge_id="kb-1",
        source_type="manuals",
        title="Version 1",
        content="Original content with term ALPHA.",
    )
    await store.store_content(
        source_id="src-up",
        knowledge_id="kb-1",
        source_type="manuals",
        title="Version 2",
        content="Updated content with term BETA.",
    )
    results_alpha = await store.search_content(query="ALPHA", knowledge_id="kb-1")
    results_beta = await store.search_content(query="BETA", knowledge_id="kb-1")
    assert len(results_alpha) == 0
    assert len(results_beta) == 1


def test_search_postgres_uses_core_expressions() -> None:
    """Regression: _search_postgres must not assemble SQL via text(f'...')."""
    source = inspect.getsource(PostgresDocumentStore._search_postgres)
    assert "text(f" not in source, "_search_postgres must not use text(f'...') f-string SQL assembly"
    assert "where_sql" not in source, "_search_postgres must not use the old f-string where_sql variable"
    assert "ilike_where_sql" not in source, "_search_postgres must not use the old f-string ilike_where_sql variable"


def test_postgres_headline_options_configurable() -> None:
    store = PostgresDocumentStore(
        url="sqlite+aiosqlite:///:memory:",
        headline_max_words=50,
        headline_min_words=20,
        headline_max_fragments=1,
    )
    assert store._headline_max_words == 50
    assert store._headline_min_words == 20
    assert store._headline_max_fragments == 1
