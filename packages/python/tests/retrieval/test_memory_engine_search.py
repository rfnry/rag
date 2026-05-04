from __future__ import annotations

from types import SimpleNamespace

import pytest

from rfnry_knowledge.memory.engine import MemoryEngine


async def test_search_validates_inputs(memory_cfg_factory) -> None:
    cfg = memory_cfg_factory()
    async with MemoryEngine(cfg) as engine:
        with pytest.raises(ValueError):
            await engine.search("", memory_id="u")
        with pytest.raises(ValueError):
            await engine.search("q", memory_id=" ")


async def test_search_filters_by_memory_id(memory_cfg_factory) -> None:
    cfg = memory_cfg_factory()
    cfg.ingestion.vector_store._search_results = [
        SimpleNamespace(
            point_id="r1", score=0.9,
            payload={"memory_row_id": "r1", "memory_id": "u-7",
                     "knowledge_id": "u-7", "content": "hello", "text": "hello",
                     "text_hash": "h", "linked_memory_row_ids": []},
        ),
    ]
    async with MemoryEngine(cfg) as engine:
        results = await engine.search("hello", memory_id="u-7", top_k=5)
    assert len(results) == 1
    assert results[0].row.memory_id == "u-7"
    assert "semantic" in results[0].pillar_scores


async def test_search_returns_empty_on_no_results(memory_cfg_factory) -> None:
    cfg = memory_cfg_factory()
    cfg.ingestion.vector_store._search_results = []
    async with MemoryEngine(cfg) as engine:
        results = await engine.search("nope", memory_id="u-7")
    assert results == ()
