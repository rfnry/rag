from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from rfnry_knowledge.memory.engine import MemoryEngine
from rfnry_knowledge.memory.models import (
    ExtractedMemory,
    Interaction,
    InteractionTurn,
)


async def test_add_rejects_empty_turns(memory_cfg_factory) -> None:
    cfg = memory_cfg_factory()
    async with MemoryEngine(cfg) as engine:
        with pytest.raises(ValueError):
            await engine.add(Interaction(turns=()), memory_id="u")


async def test_add_rejects_blank_memory_id(memory_cfg_factory) -> None:
    cfg = memory_cfg_factory()
    async with MemoryEngine(cfg) as engine:
        with pytest.raises(ValueError):
            await engine.add(Interaction(turns=(InteractionTurn("u", "x"),)), memory_id="  ")


async def test_add_returns_empty_when_extractor_yields_nothing(
    memory_cfg_factory, stub_memory_extractor_factory,
) -> None:
    cfg = memory_cfg_factory(extractor=stub_memory_extractor_factory([]))
    async with MemoryEngine(cfg) as engine:
        out = await engine.add(
            Interaction(turns=(InteractionTurn("u", "x"),)), memory_id="u",
        )
    assert out == ()
    assert cfg.vector_store.points == []


async def test_add_writes_one_point_per_extracted_memory(
    memory_cfg_factory, stub_memory_extractor_factory,
) -> None:
    cfg = memory_cfg_factory(extractor=stub_memory_extractor_factory([
        ExtractedMemory(text="user lives in Lisbon", attributed_to="user"),
        ExtractedMemory(text="user uses FastAPI", attributed_to="user"),
    ]))
    async with MemoryEngine(cfg) as engine:
        rows = await engine.add(
            Interaction(turns=(InteractionTurn("u", "I moved to Lisbon."),)),
            memory_id="u-7",
        )
    assert len(rows) == 2
    assert len(cfg.vector_store.points) == 2
    payloads = [p.payload for p in cfg.vector_store.points]
    assert all(p["memory_id"] == "u-7" for p in payloads)
    assert all(p["text_hash"] for p in payloads)
    assert all("memory_row_id" in p for p in payloads)


async def test_add_dedups_against_hash_match(
    memory_cfg_factory, stub_memory_extractor_factory,
) -> None:
    cfg = memory_cfg_factory(extractor=stub_memory_extractor_factory([
        ExtractedMemory(text="user lives in Lisbon", attributed_to="user"),
    ]))
    h = hashlib.sha256(b"user lives in lisbon").hexdigest()
    cfg.vector_store._scroll_results = [
        SimpleNamespace(
            point_id="r-existing", score=0.0,
            payload={"text_hash": h, "memory_row_id": "r-existing", "memory_id": "u-7"},
        ),
    ]
    async with MemoryEngine(cfg) as engine:
        rows = await engine.add(
            Interaction(turns=(InteractionTurn("u", "I moved to Lisbon."),)),
            memory_id="u-7",
        )
    assert rows == ()
    assert cfg.vector_store.points == []


async def test_add_propagates_interaction_metadata_into_payload(
    memory_cfg_factory, stub_memory_extractor_factory,
) -> None:
    cfg = memory_cfg_factory(extractor=stub_memory_extractor_factory([
        ExtractedMemory(text="x", attributed_to=None),
    ]))
    async with MemoryEngine(cfg) as engine:
        await engine.add(
            Interaction(
                turns=(InteractionTurn("u", "x"),),
                metadata={"session_id": "abc"},
            ),
            memory_id="u",
        )
    assert cfg.vector_store.points[0].payload.get("session_id") == "abc"


async def test_add_passes_existing_memories_when_dedup_context_enabled(
    memory_cfg_factory, fake_memory_vector_store, fake_memory_embeddings,
    stub_memory_extractor_factory,
) -> None:
    """When dedup_context_top_k > 0, prior memories are fetched and passed to the extractor."""
    from rfnry_knowledge.config.memory import (
        MemoryEngineConfig,
        MemoryIngestionConfig,
        MemoryRetrievalConfig,
    )
    extractor = stub_memory_extractor_factory([])
    cfg = MemoryEngineConfig(
        ingestion=MemoryIngestionConfig(
            extractor=extractor,
            embeddings=fake_memory_embeddings,
            dedup_context_top_k=2,
        ),
        retrieval=MemoryRetrievalConfig(),
        vector_store=fake_memory_vector_store,
        provider=SimpleNamespace(name="x", model="y"),
    )
    fake_memory_vector_store._search_results = [
        SimpleNamespace(
            point_id="prior-1", score=0.9,
            payload={
                "memory_row_id": "prior-1", "memory_id": "u",
                "text": "prior fact", "content": "prior fact", "text_hash": "h1",
                "linked_memory_row_ids": [],
            },
        ),
    ]
    async with MemoryEngine(cfg) as engine:
        await engine.add(
            Interaction(turns=(InteractionTurn("u", "new turn"),)), memory_id="u",
        )
    # Extractor was called with the prior memory in existing_memories.
    assert len(extractor.calls) == 1
    _interaction, existing = extractor.calls[0]
    assert len(existing) == 1
    assert existing[0].memory_row_id == "prior-1"


async def test_add_drops_invented_link_ids_and_increments_counter(
    memory_cfg_factory, stub_memory_extractor_factory,
) -> None:
    cfg = memory_cfg_factory(extractor=stub_memory_extractor_factory([
        ExtractedMemory(
            text="x",
            attributed_to=None,
            linked_memory_row_ids=("does-not-exist",),
        ),
    ]))
    async with MemoryEngine(cfg) as engine:
        rows = await engine.add(
            Interaction(turns=(InteractionTurn("u", "x"),)), memory_id="u",
        )
    assert len(rows) == 1
    assert rows[0].linked_memory_row_ids == ()
