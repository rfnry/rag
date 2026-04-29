"""R2.1 — RaptorTreeRegistry CRUD over the rag_raptor_trees table.

The registry is the per-knowledge_id pointer to the active tree; retrieval
(R2.3) reads it before searching summary vectors. Returning ``None`` for an
absent record is the supported contract — no tree built yet, fall through
to chunk-level retrieval. ``set_active`` is the blue/green swap mechanism
the builder (R2.2) calls after the new tree's vectors are persisted.

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
    # the prior pointer (the blue/green swap path the R2.2 builder relies on).
    store = SQLAlchemyMetadataStore(f"sqlite:///{tmp_path}/m.db")
    await store.initialize()
    registry = RaptorTreeRegistry(store)

    await registry.set_active("kid-x", "t1", [10, 1], 0.05)
    await registry.set_active("kid-x", "t2", [20, 2, 1], 0.10)

    assert await registry.get_active("kid-x") == "t2"
