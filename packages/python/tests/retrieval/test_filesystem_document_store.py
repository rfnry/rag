import pytest

from rfnry_rag.stores.document.filesystem import FilesystemDocumentStore


@pytest.fixture
async def store(tmp_path):
    s = FilesystemDocumentStore(str(tmp_path / "knowledge"))
    await s.initialize()
    yield s
    await s.shutdown()


async def test_store_creates_file(store, tmp_path):
    await store.store_content(
        source_id="src-001",
        knowledge_id="kb-1",
        source_type="manuals",
        title="Pump Manual",
        content="The FBD-20254-MERV13 filter specs.",
    )
    expected = tmp_path / "knowledge" / "kb-1" / "manuals" / "src-001.md"
    assert expected.exists()
    text = expected.read_text()
    assert "FBD-20254-MERV13" in text
    assert '"Pump Manual"' in text


async def test_search_exact_match(store):
    await store.store_content(
        source_id="src-001",
        knowledge_id="kb-1",
        source_type="manuals",
        title="Manual",
        content="The FBD-20254-MERV13 filter has a pressure drop of 0.25 inches.",
    )
    results = await store.search_content(query="FBD-20254", knowledge_id="kb-1")
    assert len(results) >= 1
    assert results[0].source_id == "src-001"
    assert "FBD-20254" in results[0].excerpt


async def test_search_ranked_bm25(store):
    await store.store_content(
        source_id="src-a",
        knowledge_id="kb-1",
        source_type="manuals",
        title="Doc A",
        content="Filter filter filter specs and filter details.",
    )
    await store.store_content(
        source_id="src-b",
        knowledge_id="kb-1",
        source_type="manuals",
        title="Doc B",
        content="Unrelated content about pumps.",
    )
    results = await store.search_content(query="filter", knowledge_id="kb-1")
    assert len(results) >= 1
    assert results[0].source_id == "src-a"


async def test_search_scoped_by_knowledge_id(store):
    await store.store_content(
        source_id="src-a", knowledge_id="kb-1", source_type="manuals", title="Doc A", content="ABC-123 specs."
    )
    await store.store_content(
        source_id="src-b", knowledge_id="kb-2", source_type="manuals", title="Doc B", content="ABC-123 other specs."
    )
    results = await store.search_content(query="ABC-123", knowledge_id="kb-1")
    assert len(results) == 1
    assert results[0].source_id == "src-a"


async def test_search_scoped_by_source_type(store):
    await store.store_content(
        source_id="src-m", knowledge_id="kb-1", source_type="manuals", title="Manual", content="Part XYZ-789 manual."
    )
    await store.store_content(
        source_id="src-d",
        knowledge_id="kb-1",
        source_type="drawings",
        title="Drawing",
        content="Part XYZ-789 drawing.",
    )
    results = await store.search_content(query="XYZ-789", knowledge_id="kb-1", source_type="manuals")
    assert len(results) == 1
    assert results[0].source_id == "src-m"


async def test_search_empty(store):
    results = await store.search_content(query="nonexistent", knowledge_id="kb-1")
    assert results == []


async def test_delete_content(store, tmp_path):
    await store.store_content(
        source_id="src-del",
        knowledge_id="kb-1",
        source_type="manuals",
        title="To Delete",
        content="Unique searchable content.",
    )
    await store.delete_content("src-del")
    results = await store.search_content(query="Unique searchable", knowledge_id="kb-1")
    assert results == []
    expected = tmp_path / "knowledge" / "kb-1" / "manuals" / "src-del.md"
    assert not expected.exists()


async def test_store_content_upsert(store):
    await store.store_content(
        source_id="src-up", knowledge_id="kb-1", source_type="manuals", title="V1", content="Original with ALPHA."
    )
    await store.store_content(
        source_id="src-up", knowledge_id="kb-1", source_type="manuals", title="V2", content="Updated with BETA."
    )
    results_alpha = await store.search_content(query="ALPHA", knowledge_id="kb-1")
    results_beta = await store.search_content(query="BETA", knowledge_id="kb-1")
    assert len(results_alpha) == 0
    assert len(results_beta) == 1


async def test_no_knowledge_id(store, tmp_path):
    await store.store_content(
        source_id="src-none",
        knowledge_id=None,
        source_type="manuals",
        title="No KB",
        content="Content without knowledge id.",
    )
    expected = tmp_path / "knowledge" / "_default" / "manuals" / "src-none.md"
    assert expected.exists()
    results = await store.search_content(query="without knowledge", knowledge_id=None)
    assert len(results) >= 1


@pytest.mark.parametrize("bad_id", ["../etc", "../../outside", "/abs/path", "a/b/c", "..", ".", "  ", ""])
async def test_rejects_traversal_in_knowledge_id(store, bad_id):
    with pytest.raises(ValueError, match="invalid.*component"):
        await store.store_content(
            source_id="src-bad",
            knowledge_id=bad_id,
            source_type="manuals",
            title="x",
            content="x",
        )


@pytest.mark.parametrize("bad_type", ["../etc", "cron.d/", "a/b", "..", ".", "  ", ""])
async def test_rejects_traversal_in_source_type(store, bad_type):
    with pytest.raises(ValueError, match="invalid.*component"):
        await store.store_content(
            source_id="src-bad",
            knowledge_id="kb",
            source_type=bad_type,
            title="x",
            content="x",
        )


@pytest.mark.parametrize("bad_id", ["../escape", "a/b"])
async def test_search_rejects_traversal_in_knowledge_id(store, bad_id):
    with pytest.raises(ValueError, match="invalid.*component"):
        await store.search_content(query="anything", knowledge_id=bad_id)


async def test_filesystem_store_handles_title_with_frontmatter_delimiter(tmp_path) -> None:
    store = FilesystemDocumentStore(base_path=str(tmp_path))
    await store.initialize()
    bad_title = "normal start\n---\nend: injected"
    await store.store_content(
        source_id="s1",
        knowledge_id=None,
        source_type=None,
        title=bad_title,
        content="hello",
    )
    hits = await store.search_content("hello")
    assert len(hits) == 1
    assert hits[0].title == bad_title
