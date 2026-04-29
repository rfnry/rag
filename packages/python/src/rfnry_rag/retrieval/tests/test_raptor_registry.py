"""RaptorTreeRegistry CRUD over the rag_raptor_trees table.

The registry is the per-knowledge_id pointer to the active tree; retrieval
reads it before searching summary vectors. Returning ``None`` for an absent
record is the supported contract — no tree built yet, fall through to
chunk-level retrieval. ``set_active`` is the blue/green swap mechanism the
builder calls after the new tree's vectors are persisted.

Tests use a real ``SQLAlchemyMetadataStore`` against a SQLite file in a
``tmp_path`` directory — mirrors the existing metadata-store test patterns
(see ``test_metadata_store_lifecycle.py``).
"""

from __future__ import annotations

import pytest

from rfnry_rag.retrieval.modules.ingestion.methods.raptor.registry import RaptorTreeRegistry
from rfnry_rag.retrieval.stores.metadata.sqlalchemy import SQLAlchemyMetadataStore


@pytest.mark.asyncio
async def test_registry_get_active_returns_none_when_unset(tmp_path) -> None:
    store = SQLAlchemyMetadataStore(f"sqlite:///{tmp_path}/m.db")
    await store.initialize()
    registry = RaptorTreeRegistry(store)

    # No row written for this knowledge_id, registry returns None.
    assert await registry.get_active("kid-x") is None


@pytest.mark.asyncio
async def test_registry_set_then_get_returns_tree_id(tmp_path) -> None:
    store = SQLAlchemyMetadataStore(f"sqlite:///{tmp_path}/m.db")
    await store.initialize()
    registry = RaptorTreeRegistry(store)

    await registry.set_active("kid-x", "t1", [10, 1], 0.05)
    assert await registry.get_active("kid-x") == "t1"


@pytest.mark.asyncio
async def test_registry_set_replaces_active_tree_id(tmp_path) -> None:
    # Upsert semantics: a second set_active for the same knowledge_id replaces
    # the prior pointer (the blue/green swap path the builder relies on).
    store = SQLAlchemyMetadataStore(f"sqlite:///{tmp_path}/m.db")
    await store.initialize()
    registry = RaptorTreeRegistry(store)

    await registry.set_active("kid-x", "t1", [10, 1], 0.05)
    await registry.set_active("kid-x", "t2", [20, 2, 1], 0.10)

    assert await registry.get_active("kid-x") == "t2"


@pytest.mark.asyncio
async def test_set_active_persists_level_counts_json_roundtrip(tmp_path) -> None:
    """Regression guard: ``level_counts_json`` survives the JSON encode/decode trip.

    The builder reads this column back for GC and observability — if the
    encoding contract drifts (e.g. someone swaps to the SQLAlchemy ``JSON``
    type or forgets the ``json.dumps`` call), the builder would be the first
    to find out. Reaching directly through to ``_session_factory`` keeps the
    registry's public API surface unchanged.
    """
    import json

    from rfnry_rag.retrieval.stores.metadata.sqlalchemy import _RaptorTreeRow  # noqa: SLF001

    store = SQLAlchemyMetadataStore(f"sqlite:///{tmp_path}/m.db")
    await store.initialize()
    registry = RaptorTreeRegistry(store)

    await registry.set_active(
        knowledge_id="kid-x",
        tree_id="t1",
        level_counts=[1000, 100, 11, 1],
        cost_usd=0.05,
    )

    # Direct DB read to verify the JSON-encoded payload survives the round-trip.
    async with store._session_factory() as session:  # noqa: SLF001
        row = await session.get(_RaptorTreeRow, "kid-x")
        assert row is not None
        assert json.loads(row.level_counts_json) == [1000, 100, 11, 1]
        assert row.total_cost_usd == 0.05
        assert row.active_tree_id == "t1"


@pytest.mark.asyncio
async def test_delete_record_is_idempotent_on_missing(tmp_path) -> None:
    """Calling delete on a never-set kid (or twice) doesn't raise.

    The GC pass leans on this — a transient failure mid-cleanup must be
    safe to retry without surfacing as a hard error.
    """
    store = SQLAlchemyMetadataStore(f"sqlite:///{tmp_path}/m.db")
    await store.initialize()
    registry = RaptorTreeRegistry(store)

    # No prior set; this should be a no-op.
    await registry.delete_record("kid-never-existed")

    # Set then delete twice.
    await registry.set_active("kid-x", "t1", [10, 1], None)
    await registry.delete_record("kid-x")
    await registry.delete_record("kid-x")  # second call must not raise

    assert await registry.get_active("kid-x") is None


@pytest.mark.asyncio
async def test_get_stale_trees_returns_kids_not_in_active_set(tmp_path) -> None:
    """``get_stale_trees`` finds registry rows whose knowledge_id is no longer active.

    Used by the GC pass after a knowledge_id is removed from the
    ``KnowledgeManager`` — the registry row outlives the knowledge it
    pointed at. The method returns *knowledge_ids whose row is NOT in the
    active set*, not historical tree_ids within a knowledge_id (the
    registry only tracks the active pointer per kid).
    """
    store = SQLAlchemyMetadataStore(f"sqlite:///{tmp_path}/m.db")
    await store.initialize()
    registry = RaptorTreeRegistry(store)

    await registry.set_active("kid-a", "t1", [10, 1], None)
    await registry.set_active("kid-b", "t2", [20, 2], None)
    await registry.set_active("kid-c", "t3", [30, 3], None)

    # Only kid-a is "active" in the live set; kid-b and kid-c are orphans.
    stale = await registry.get_stale_trees(active_knowledge_ids={"kid-a"})

    assert sorted(stale) == ["kid-b", "kid-c"]
