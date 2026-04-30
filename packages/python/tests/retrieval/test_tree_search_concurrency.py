import asyncio
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

from rfnry_rag.retrieval.server import RagEngine, TreeSearchConfig


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

    source_ids = [f"s{i}" for i in range(4)]
    metadata_store = MagicMock()
    metadata_store.list_source_ids = AsyncMock(return_value=source_ids)
    metadata_store.get_tree_indexes = AsyncMock(return_value={sid: tree_json for sid in source_ids})

    rag = RagEngine.__new__(RagEngine)
    rag._tree_search_service = tree_service
    rag._config = cast(
        Any,
        SimpleNamespace(
            persistence=SimpleNamespace(metadata_store=metadata_store),
            tree_search=TreeSearchConfig(enabled=False),  # default max_sources_per_query=50
        ),
    )

    await rag._run_tree_search(query="q", knowledge_id=None)

    assert max_concurrent >= 2, f"tree search ran serially (max_concurrent={max_concurrent})"


async def test_run_tree_search_caps_source_count() -> None:
    """_run_tree_search must limit fan-out to max_sources_per_query."""
    search_call_count = 0

    async def counting_search(*args, **kwargs):
        nonlocal search_call_count
        search_call_count += 1
        return []

    tree_service = MagicMock()
    tree_service.search = AsyncMock(side_effect=counting_search)
    tree_service.to_retrieved_chunks = MagicMock(return_value=[])

    tree_json = (
        '{"source_id": "s0", "doc_name": "Doc", "doc_description": null,'
        ' "structure": [], "page_count": 1, "created_at": "2024-01-01T00:00:00",'
        ' "pages": [{"index": 0, "text": "x", "token_count": 1}]}'
    )

    # 200 sources; cap should be applied so only 5 are searched
    all_ids = [f"s{i}" for i in range(200)]
    capped_ids = all_ids[:5]
    metadata_store = MagicMock()
    metadata_store.list_source_ids = AsyncMock(return_value=all_ids)
    metadata_store.get_tree_indexes = AsyncMock(return_value={sid: tree_json for sid in capped_ids})

    rag = RagEngine.__new__(RagEngine)
    rag._tree_search_service = tree_service
    rag._config = cast(
        Any,
        SimpleNamespace(
            persistence=SimpleNamespace(metadata_store=metadata_store),
            tree_search=TreeSearchConfig(enabled=False, max_sources_per_query=5),
        ),
    )

    await rag._run_tree_search(query="q", knowledge_id=None)

    assert search_call_count <= 5, f"tree search fanned out to {search_call_count} sources; expected at most 5"
