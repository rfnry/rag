"""RaptorTreeRegistry — async CRUD over the ``rag_raptor_trees`` table.

R2.1 ships the registry fully implemented because retrieval (R2.3) needs to
ask "is there an active tree for this knowledge_id?" before searching summary
vectors. Returning ``None`` for an absent record is the supported contract —
no tree built yet, fall through to chunk-level retrieval. The builder (R2.2)
writes through ``set_active`` after a successful blue/green swap.

The registry is a thin façade over ``SQLAlchemyMetadataStore``'s engine and
session factory; it intentionally does NOT add a parallel ``BaseMetadataStore``
method per the "sibling, not fold-in" convention from the R2.1 plan.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import delete, select

from rfnry_rag.retrieval.common.logging import get_logger
from rfnry_rag.retrieval.stores.metadata.sqlalchemy import (
    SQLAlchemyMetadataStore,
    _RaptorTreeRow,
)

logger = get_logger(__name__)


class RaptorTreeRegistry:
    """Per-knowledge_id pointer to the currently-active RAPTOR tree.

    Operations are point-lookups by ``knowledge_id`` (the primary key);
    upserts replace the active pointer atomically; deletes are idempotent
    (no-op when the row is absent). The store is dependency-injected so
    consumers reuse one connection pool.
    """

    def __init__(self, store: SQLAlchemyMetadataStore) -> None:
        # Reach through to the private session factory + engine. Adding a
        # public accessor would broaden the metadata-store API surface for a
        # one-consumer dependency; the registry is conceptually part of the
        # metadata store and lives in the same SDK boundary.
        self._session_factory = store._session_factory  # noqa: SLF001
        self._engine = store._engine  # noqa: SLF001

    async def get_active(self, knowledge_id: str) -> str | None:
        """Return the active tree id for ``knowledge_id``, or ``None``.

        ``None`` is the "no tree built yet" signal — retrieval should fall
        through to chunk-level search instead of raising.
        """
        async with self._session_factory() as session:
            row = await session.get(_RaptorTreeRow, knowledge_id)
            if row is None:
                return None
            return row.active_tree_id

    async def set_active(
        self,
        knowledge_id: str,
        tree_id: str,
        level_counts: list[int],
        cost_usd: float | None,
    ) -> None:
        """Upsert the active tree pointer for ``knowledge_id``.

        ``level_counts`` is JSON-serialised on write so the column stays
        portable across SQLite + PostgreSQL without a JSON column type.
        Replacing the pointer is the documented blue/green swap mechanism:
        callers should ensure the new ``tree_id``'s vectors are persisted
        before calling ``set_active``, then GC the prior id afterwards.
        """
        values: dict[str, Any] = dict(
            knowledge_id=knowledge_id,
            active_tree_id=tree_id,
            built_at=datetime.now(UTC),
            level_counts_json=json.dumps(level_counts),
            total_cost_usd=cost_usd,
        )
        dialect = self._engine.dialect.name
        async with self._session_factory() as session:
            upsert_stmt: Any
            if dialect == "sqlite":
                from sqlalchemy.dialects.sqlite import insert as sqlite_insert

                upsert_stmt = sqlite_insert(_RaptorTreeRow).values(**values)
                upsert_stmt = upsert_stmt.on_conflict_do_update(
                    index_elements=["knowledge_id"],
                    set_={
                        "active_tree_id": upsert_stmt.excluded.active_tree_id,
                        "built_at": upsert_stmt.excluded.built_at,
                        "level_counts_json": upsert_stmt.excluded.level_counts_json,
                        "total_cost_usd": upsert_stmt.excluded.total_cost_usd,
                    },
                )
            elif dialect == "postgresql":
                from sqlalchemy.dialects.postgresql import insert as pg_insert

                upsert_stmt = pg_insert(_RaptorTreeRow).values(**values)
                upsert_stmt = upsert_stmt.on_conflict_do_update(
                    index_elements=["knowledge_id"],
                    set_={
                        "active_tree_id": upsert_stmt.excluded.active_tree_id,
                        "built_at": upsert_stmt.excluded.built_at,
                        "level_counts_json": upsert_stmt.excluded.level_counts_json,
                        "total_cost_usd": upsert_stmt.excluded.total_cost_usd,
                    },
                )
            else:
                raise NotImplementedError(
                    f"RaptorTreeRegistry.set_active not implemented for dialect {dialect!r}"
                )
            await session.execute(upsert_stmt)
            await session.commit()

    async def delete_record(self, knowledge_id: str) -> None:
        """Remove the registry row for ``knowledge_id``. Idempotent.

        Called by the knowledge-manager removal flow (not by R2.1 itself —
        the method is registered now so R2.2 / future cleanup tooling has a
        stable hook).
        """
        async with self._session_factory() as session:
            await session.execute(
                delete(_RaptorTreeRow).where(_RaptorTreeRow.knowledge_id == knowledge_id)
            )
            await session.commit()

    async def get_stale_trees(self, active_knowledge_ids: set[str]) -> list[str]:
        """Return knowledge_ids whose row exists but is not in ``active_knowledge_ids``.

        Note:
            The registry only tracks the active tree_id per knowledge_id (not
            historical tree_ids), so this method finds orphan registry rows by
            knowledge_id — not stale tree_ids within a knowledge_id.
            Vector-store-level stale-tree GC (deleting old summary vectors after
            a blue/green swap) is separate; see ``RaptorTreeBuilder``'s
            swap-and-GC step (R2.2).

        Used by the GC pass after a knowledge_id is removed from the
        ``KnowledgeManager`` — the registry row may have outlived the
        knowledge it pointed at. Caller iterates and calls ``delete_record``
        plus the relevant vector-store cleanup.
        """
        async with self._session_factory() as session:
            rows = (await session.execute(select(_RaptorTreeRow.knowledge_id))).scalars().all()
        return [kid for kid in rows if kid not in active_knowledge_ids]
