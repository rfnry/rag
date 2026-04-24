import asyncio
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

from rfnry_rag.retrieval.server import RagEngine


async def test_run_tree_search_fans_out_sources_concurrently() -> None:
    """_run_tree_search must gather across sources — serial awaits cause p99 spikes."""
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
    metadata_store.list_sources = AsyncMock(
        return_value=[SimpleNamespace(source_id=f"s{i}") for i in range(4)]
    )
    metadata_store.get_tree_index = AsyncMock(return_value=tree_json)

    rag = RagEngine.__new__(RagEngine)
    rag._tree_search_service = tree_service
    rag._config = cast(Any, SimpleNamespace(persistence=SimpleNamespace(metadata_store=metadata_store)))

    await rag._run_tree_search(query="q", knowledge_id=None)

    assert max_concurrent >= 2, f"tree search ran serially (max_concurrent={max_concurrent})"
