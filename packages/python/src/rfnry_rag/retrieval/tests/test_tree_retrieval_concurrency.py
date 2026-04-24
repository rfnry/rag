import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from rfnry_rag.retrieval.modules.retrieval.methods.tree import TreeRetrieval


async def test_tree_retrieval_fans_out_sources_concurrently() -> None:
    """TreeRetrieval.search must gather per-source work, not loop serially."""
    concurrent = 0
    max_concurrent = 0

    async def slow_search(*args, **kwargs):
        nonlocal concurrent, max_concurrent
        concurrent += 1
        max_concurrent = max(max_concurrent, concurrent)
        await asyncio.sleep(0.02)
        concurrent -= 1
        return []

    tree_service = MagicMock()
    tree_service.search = AsyncMock(side_effect=slow_search)
    tree_service.to_retrieved_chunks = MagicMock(return_value=[])

    # TreeIndex.from_dict requires: source_id, doc_name, structure, page_count, created_at
    tree_json = (
        '{"source_id": "s0", "doc_name": "Doc", "doc_description": null,'
        ' "structure": [], "page_count": 1, "created_at": "2024-01-01T00:00:00",'
        ' "pages": [{"index": 0, "text": "x", "token_count": 1}]}'
    )

    metadata_store = MagicMock()
    metadata_store.list_sources = AsyncMock(return_value=[SimpleNamespace(source_id=f"s{i}") for i in range(4)])
    metadata_store.get_tree_index = AsyncMock(return_value=tree_json)

    retrieval = TreeRetrieval(service=tree_service, metadata_store=metadata_store)

    await retrieval.search(query="q", top_k=10, filters=None, knowledge_id=None)

    assert max_concurrent >= 2, f"inner tree loop ran serially (max_concurrent={max_concurrent})"
