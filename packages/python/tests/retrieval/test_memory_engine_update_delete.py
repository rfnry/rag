from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from rfnry_knowledge.exceptions import MemoryNotFoundError
from rfnry_knowledge.memory.engine import MemoryEngine


async def test_update_raises_when_row_missing(memory_cfg_factory) -> None:
    cfg = memory_cfg_factory()
    cfg.ingestion.vector_store._scroll_results = []
    async with MemoryEngine(cfg) as engine:
        with pytest.raises(MemoryNotFoundError):
            await engine.update("missing-id", "new text", memory_id="u")


async def test_update_overwrites_text_in_place(memory_cfg_factory) -> None:
    cfg = memory_cfg_factory()
    cfg.ingestion.vector_store._scroll_results = [SimpleNamespace(
        point_id="r1", score=0.0,
        payload={
            "memory_row_id": "r1", "memory_id": "u", "knowledge_id": "u",
            "text": "old", "content": "old", "text_hash": "h",
            "attributed_to": None, "linked_memory_row_ids": [],
            "created_at": datetime.now(UTC).isoformat(),
        },
    )]
    async with MemoryEngine(cfg) as engine:
        after = await engine.update("r1", "new text", memory_id="u")
    assert after.text == "new text"
    assert after.text_hash != "h"
    assert any(p.point_id == "r1" for p in cfg.ingestion.vector_store.points)


async def test_delete_raises_when_missing(memory_cfg_factory) -> None:
    cfg = memory_cfg_factory()
    cfg.ingestion.vector_store._scroll_results = []
    async with MemoryEngine(cfg) as engine:
        with pytest.raises(MemoryNotFoundError):
            await engine.delete("missing", memory_id="u")


async def test_delete_drops_from_vector_store(memory_cfg_factory) -> None:
    cfg = memory_cfg_factory()
    cfg.ingestion.vector_store._scroll_results = [SimpleNamespace(
        point_id="r1", score=0.0,
        payload={"memory_row_id": "r1", "memory_id": "u", "knowledge_id": "u",
                 "text": "x", "content": "x", "text_hash": "h",
                 "linked_memory_row_ids": [], "created_at": datetime.now(UTC).isoformat()},
    )]
    async with MemoryEngine(cfg) as engine:
        await engine.delete("r1", memory_id="u")
    assert {"memory_row_id": "r1"} in cfg.ingestion.vector_store.deleted


async def test_update_writes_to_document_store_when_postgres_fts(
    fake_memory_vector_store, fake_memory_embeddings, stub_memory_extractor_factory,
) -> None:
    from unittest.mock import AsyncMock

    from rfnry_knowledge.config.memory import (
        MemoryEngineConfig,
        MemoryIngestionConfig,
        MemoryRetrievalConfig,
    )

    fake_doc = AsyncMock()

    cfg = MemoryEngineConfig(
        ingestion=MemoryIngestionConfig(
            extractor=stub_memory_extractor_factory(),
            embeddings=fake_memory_embeddings,
            vector_store=fake_memory_vector_store,
            document_store=fake_doc,
            keyword_backend="postgres_fts",
        ),
        retrieval=MemoryRetrievalConfig(),
    )
    fake_memory_vector_store._scroll_results = [SimpleNamespace(
        point_id="r1", score=0.0,
        payload={
            "memory_row_id": "r1", "memory_id": "u", "knowledge_id": "u",
            "text": "old", "content": "old", "text_hash": "h",
            "attributed_to": None, "linked_memory_row_ids": [],
            "created_at": datetime.now(UTC).isoformat(),
        },
    )]
    async with MemoryEngine(cfg) as engine:
        await engine.update("r1", "new text", memory_id="u")
    fake_doc.store_content.assert_awaited_once()
    kwargs = fake_doc.store_content.await_args.kwargs
    assert kwargs["source_id"] == "r1"
    assert kwargs["content"] == "new text"
