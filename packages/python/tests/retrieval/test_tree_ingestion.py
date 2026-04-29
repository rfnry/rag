from unittest.mock import AsyncMock, MagicMock

from rfnry_rag.retrieval.modules.ingestion.methods.tree import TreeIngestion


async def test_ingest_with_pages():
    tree_service = AsyncMock()
    tree_service.build_tree_index = AsyncMock(return_value=MagicMock())
    tree_service.save_tree_index = AsyncMock()
    method = TreeIngestion(tree_service=tree_service)
    assert method.name == "tree"
    pages = [MagicMock(page_number=1, content="Page 1 text")]
    await method.ingest(
        source_id="src-1",
        knowledge_id=None,
        source_type=None,
        source_weight=1.0,
        title="Test Doc",
        full_text="Page 1 text",
        chunks=[],
        tags=[],
        metadata={},
        pages=pages,
    )
    tree_service.build_tree_index.assert_called_once()
    tree_service.save_tree_index.assert_called_once()


async def test_ingest_without_pages_skips():
    tree_service = AsyncMock()
    method = TreeIngestion(tree_service=tree_service)
    await method.ingest(
        source_id="src-1",
        knowledge_id=None,
        source_type=None,
        source_weight=1.0,
        title="Test",
        full_text="text",
        chunks=[],
        tags=[],
        metadata={},
        pages=None,
    )
    tree_service.build_tree_index.assert_not_called()


async def test_delete_is_noop():
    tree_service = AsyncMock()
    method = TreeIngestion(tree_service=tree_service)
    await method.delete("src-1")
