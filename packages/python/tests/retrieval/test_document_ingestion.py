from unittest.mock import AsyncMock

from rfnry_knowledge.ingestion.methods.keyword import KeywordIngestion


async def test_ingest_stores_content():
    store = AsyncMock()
    store.store_content = AsyncMock()
    method = KeywordIngestion(store=store)
    assert method.name == "keyword"
    await method.ingest(
        source_id="src-1",
        knowledge_id="kb-1",
        source_type="manuals",
        source_weight=1.0,
        title="Test Doc",
        full_text="Full document text here.",
        chunks=[],
        tags=[],
        metadata={},
    )
    store.store_content.assert_called_once_with(
        source_id="src-1",
        knowledge_id="kb-1",
        source_type="manuals",
        title="Test Doc",
        content="Full document text here.",
    )


async def test_delete():
    store = AsyncMock()
    store.delete_content = AsyncMock()
    method = KeywordIngestion(store=store)
    await method.delete("src-1")
    store.delete_content.assert_called_once_with("src-1")
