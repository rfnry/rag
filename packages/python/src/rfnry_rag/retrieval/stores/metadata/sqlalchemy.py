import json
from datetime import datetime
from typing import Any

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    delete,
    inspect,
    select,
    text,
    update,
)
from sqlalchemy.engine import make_url
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.schema import ColumnDefault

from rfnry_rag.retrieval.common.errors import ConfigurationError, DuplicateSourceError, SourceNotFoundError
from rfnry_rag.retrieval.common.logging import get_logger
from rfnry_rag.retrieval.common.models import Source, SourceStats

logger = get_logger(__name__)

# Current schema version. Bump on every additive migration so we can detect
# downgrade attempts and avoid double-applying ALTER statements under concurrent
# process start.
_SCHEMA_VERSION = 2


class _Base(DeclarativeBase):
    pass


class _SchemaMeta(_Base):
    __tablename__ = "rag_schema_meta"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[int] = mapped_column(Integer, nullable=False)


class _SourceRow(_Base):
    __tablename__ = "rag_sources"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    knowledge_id: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    source_type: Mapped[str | None] = mapped_column(String(100), nullable=True)
    source_weight: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="completed")
    metadata_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    tags_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    embedding_model: Mapped[str] = mapped_column(String(100), nullable=False)
    file_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    stale: Mapped[bool] = mapped_column(nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    tree_index_json: Mapped[str | None] = mapped_column(Text, nullable=True)


class _SourceStatsRow(_Base):
    __tablename__ = "rag_source_stats"

    source_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("rag_sources.id", ondelete="CASCADE"), primary_key=True
    )
    total_chunks: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_pages: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    avg_chunk_size: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    processing_time: Mapped[float] = mapped_column(Float, nullable=False, default=0)
    total_hits: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    grounded_hits: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    ungrounded_hits: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class SQLAlchemyMetadataStore:
    def __init__(
        self,
        url: str,
        *,
        pool_size: int | None = None,
        max_overflow: int | None = None,
        pool_recycle: int = 1800,
        pool_pre_ping: bool = True,
        pool_timeout: int = 10,
        echo: bool = False,
    ) -> None:
        if pool_timeout <= 0:
            raise ConfigurationError(f"pool_timeout must be > 0, got {pool_timeout}")
        # SQLAlchemy interprets -1 as "never recycle"; reject 0 or other negatives.
        if pool_recycle != -1 and pool_recycle <= 0:
            raise ConfigurationError(f"pool_recycle must be > 0 or -1 (disable), got {pool_recycle}")

        parsed = make_url(url)
        if parsed.drivername == "postgresql":
            parsed = parsed.set(drivername="postgresql+asyncpg")
        elif parsed.drivername == "sqlite":
            parsed = parsed.set(drivername="sqlite+aiosqlite")

        kwargs: dict[str, Any] = {"echo": echo}
        # Pool tuning is only meaningful for pooled drivers (e.g. postgresql+asyncpg).
        # SQLite (aiosqlite) uses StaticPool and rejects pool_size/max_overflow.
        if parsed.drivername.startswith("postgresql"):
            kwargs["pool_pre_ping"] = pool_pre_ping
            kwargs["pool_recycle"] = pool_recycle
            # pool_timeout default in SQLAlchemy is 30s; lower it so pool
            # exhaustion surfaces as a fast error instead of a hidden stall.
            kwargs["pool_timeout"] = pool_timeout
            if pool_size is not None:
                kwargs["pool_size"] = pool_size
            if max_overflow is not None:
                kwargs["max_overflow"] = max_overflow

        self._engine = create_async_engine(parsed, **kwargs)
        self._session_factory = async_sessionmaker(self._engine, class_=AsyncSession, expire_on_commit=False)

    async def initialize(self) -> None:
        driver = self._engine.dialect.name
        if driver == "sqlite":
            logger.info(
                "sqlalchemy metadata store initializing: driver=sqlite "
                "static pool (pool_size/max_overflow not applicable)"
            )
        else:
            pool = self._engine.sync_engine.pool
            pool_size = pool.size() if hasattr(pool, "size") else "n/a"
            overflow = getattr(pool, "_max_overflow", "n/a")
            recycle = getattr(pool, "_recycle", "n/a")
            timeout = getattr(pool, "_timeout", "n/a")
            logger.info(
                "sqlalchemy metadata store initializing: driver=%s "
                "pool_size=%s max_overflow=%s pool_recycle=%ss pool_timeout=%ss",
                driver,
                pool_size,
                overflow,
                recycle,
                timeout,
            )
        async with self._engine.begin() as conn:
            await conn.run_sync(_Base.metadata.create_all)
            # Check/advance schema version BEFORE running migrations so concurrent
            # processes don't double-apply ALTER statements. The version check and
            # the migration run share a single transaction.
            await conn.run_sync(self._apply_schema_migrations)
        logger.info("metadata store tables initialized (schema_version=%d)", _SCHEMA_VERSION)

    @staticmethod
    def _apply_schema_migrations(conn) -> None:
        """Read + advance schema version, then run additive migrations if needed.

        All inside the caller's transaction so concurrent initializers serialise
        via the version-row lock."""
        # Lock-insert the version row (first process only).
        conn.execute(
            text("INSERT INTO rag_schema_meta (key, value) VALUES ('schema_version', 0) ON CONFLICT DO NOTHING")
            if conn.dialect.name == "postgresql"
            else text("INSERT OR IGNORE INTO rag_schema_meta (key, value) VALUES ('schema_version', 0)")
        )
        current = conn.execute(text("SELECT value FROM rag_schema_meta WHERE key = 'schema_version'")).scalar_one()

        if current > _SCHEMA_VERSION:
            raise RuntimeError(
                f"metadata store schema_version={current} is newer than "
                f"code's _SCHEMA_VERSION={_SCHEMA_VERSION}; downgrade is not supported"
            )

        if current < _SCHEMA_VERSION:
            logger.info("migrating metadata schema: %d -> %d", current, _SCHEMA_VERSION)
            SQLAlchemyMetadataStore._migrate_missing_columns(conn)
            SQLAlchemyMetadataStore._migrate_missing_indexes(conn)
            conn.execute(
                text("UPDATE rag_schema_meta SET value = :v WHERE key = 'schema_version'"),
                {"v": _SCHEMA_VERSION},
            )
        else:
            # Still run ADD-COLUMN / ADD-INDEX for safety on first boot after upgrading
            # code when the version row was set to _SCHEMA_VERSION by a prior boot
            # that didn't actually add new columns/indexes (e.g. code downgrade + upgrade).
            SQLAlchemyMetadataStore._migrate_missing_columns(conn)
            SQLAlchemyMetadataStore._migrate_missing_indexes(conn)

    @staticmethod
    def _render_default_literal(val: object, dialect) -> str:
        """Render a Python default value as a SQL literal, dialect-safe."""
        if isinstance(val, bool):
            return "1" if val else "0"
        if val is None:
            return "NULL"
        if isinstance(val, (int, float)):
            return str(val)
        proc = String().literal_processor(dialect=dialect)
        return proc(str(val))

    @staticmethod
    def _migrate_missing_columns(conn) -> None:
        """Add any columns that exist in the model but not in the database (schema evolution)."""
        insp = inspect(conn)
        preparer = conn.dialect.identifier_preparer
        for table in _Base.metadata.sorted_tables:
            if not insp.has_table(table.name):
                continue
            existing = {col["name"] for col in insp.get_columns(table.name)}
            for column in table.columns:
                if column.name in existing:
                    continue
                col_type = column.type.compile(conn.dialect)
                default = ""
                if column.default is not None and isinstance(column.default, ColumnDefault):
                    literal = SQLAlchemyMetadataStore._render_default_literal(column.default.arg, conn.dialect)
                    default = f" DEFAULT {literal}"
                nullable = "" if column.nullable else " NOT NULL"
                safe_table = preparer.quote_identifier(table.name)
                safe_col = preparer.quote_identifier(column.name)
                sql = f"ALTER TABLE {safe_table} ADD COLUMN {safe_col} {col_type}{nullable}{default}"
                conn.execute(text(sql))
                logger.info("migrated column: %s.%s", table.name, column.name)

    @staticmethod
    def _migrate_missing_indexes(conn) -> None:
        """Add any indexes that exist in the model but not in the database (schema evolution).

        Uses CREATE INDEX IF NOT EXISTS so the statement is idempotent on both
        SQLite and PostgreSQL — safe to call on every boot."""
        # Schema version 2: index on rag_sources.file_hash for find_by_hash performance.
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_rag_sources_file_hash"
                " ON rag_sources (file_hash)"
            )
        )
        logger.info("ensured index: rag_sources.file_hash")

    async def create_source(self, source: Source) -> None:
        row = _SourceRow(
            id=source.source_id,
            knowledge_id=source.knowledge_id,
            source_type=source.source_type,
            source_weight=source.source_weight,
            status=source.status,
            metadata_json=json.dumps(source.metadata),
            tags_json=json.dumps(source.tags),
            chunk_count=source.chunk_count,
            embedding_model=source.embedding_model,
            file_hash=source.file_hash,
            stale=source.stale,
            created_at=source.created_at,
        )
        async with self._session_factory() as session:
            try:
                session.add(row)
                await session.commit()
            except IntegrityError as exc:
                await session.rollback()
                raise DuplicateSourceError(f"Source {source.source_id} already exists") from exc

    async def get_source(self, source_id: str) -> Source | None:
        async with self._session_factory() as session:
            result = await session.execute(select(_SourceRow).where(_SourceRow.id == source_id))
            row = result.scalar_one_or_none()
            if row is None:
                return None
            return self._row_to_source(row)

    async def list_sources(self, knowledge_id: str | None = None) -> list[Source]:
        async with self._session_factory() as session:
            stmt = select(_SourceRow)
            if knowledge_id is not None:
                stmt = stmt.where(_SourceRow.knowledge_id == knowledge_id)
            stmt = stmt.order_by(_SourceRow.created_at.desc())
            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [self._row_to_source(row) for row in rows]

    async def list_source_ids(self, knowledge_id: str | None = None) -> list[str]:
        stmt = select(_SourceRow.id)
        if knowledge_id is not None:
            stmt = stmt.where(_SourceRow.knowledge_id == knowledge_id)
        async with self._session_factory() as session:
            rows = (await session.execute(stmt)).scalars().all()
        return list(rows)

    async def find_by_hash(self, hash_value: str, knowledge_id: str | None) -> Source | None:
        stmt = select(_SourceRow).where(_SourceRow.file_hash == hash_value)
        if knowledge_id is not None:
            stmt = stmt.where(_SourceRow.knowledge_id == knowledge_id)
        stmt = stmt.limit(1)
        async with self._session_factory() as session:
            row = (await session.execute(stmt)).scalars().first()
            return None if row is None else self._row_to_source(row)

    _ALLOWED_UPDATE_FIELDS = {
        "metadata",
        "tags",
        "chunk_count",
        "embedding_model",
        "file_hash",
        "stale",
        "knowledge_id",
        "source_type",
        "source_weight",
        "status",
        "tree_index_json",
    }

    async def update_source(self, source_id: str, **fields) -> None:
        unknown = set(fields.keys()) - self._ALLOWED_UPDATE_FIELDS
        if unknown:
            raise ValueError(f"Unknown fields for source update: {unknown}")

        update_values = {}
        if "metadata" in fields:
            update_values["metadata_json"] = json.dumps(fields.pop("metadata"))
        if "tags" in fields:
            update_values["tags_json"] = json.dumps(fields.pop("tags"))
        update_values.update(fields)

        if not update_values:
            return

        async with self._session_factory() as session:
            cursor_result = await session.execute(
                update(_SourceRow).where(_SourceRow.id == source_id).values(**update_values)
            )
            if cursor_result.rowcount == 0:  # type: ignore[attr-defined]
                raise SourceNotFoundError(f"Source {source_id} not found")
            await session.commit()

    async def delete_source(self, source_id: str) -> None:
        async with self._session_factory() as session:
            await session.execute(delete(_SourceStatsRow).where(_SourceStatsRow.source_id == source_id))
            await session.execute(delete(_SourceRow).where(_SourceRow.id == source_id))
            await session.commit()

    async def record_hit(self, source_id: str, chunk_id: str, grounded: bool) -> None:
        async with self._session_factory() as session:
            result = await session.execute(
                select(_SourceStatsRow).where(_SourceStatsRow.source_id == source_id).with_for_update()
            )
            stats_row = result.scalar_one_or_none()

            if stats_row is None:
                try:
                    stats_row = _SourceStatsRow(
                        source_id=source_id,
                        total_hits=1,
                        grounded_hits=1 if grounded else 0,
                        ungrounded_hits=0 if grounded else 1,
                    )
                    session.add(stats_row)
                    await session.commit()
                except IntegrityError:
                    await session.rollback()
                    result = await session.execute(
                        select(_SourceStatsRow).where(_SourceStatsRow.source_id == source_id).with_for_update()
                    )
                    stats_row = result.scalar_one_or_none()
                    if stats_row is not None:
                        stats_row.total_hits += 1
                        if grounded:
                            stats_row.grounded_hits += 1
                        else:
                            stats_row.ungrounded_hits += 1
                        await session.commit()
            else:
                stats_row.total_hits += 1
                if grounded:
                    stats_row.grounded_hits += 1
                else:
                    stats_row.ungrounded_hits += 1
                await session.commit()

    async def get_source_stats(self, source_id: str) -> SourceStats | None:
        async with self._session_factory() as session:
            result = await session.execute(select(_SourceStatsRow).where(_SourceStatsRow.source_id == source_id))
            row = result.scalar_one_or_none()
            if row is None:
                return None
            return SourceStats(
                source_id=row.source_id,
                total_chunks=row.total_chunks,
                total_pages=row.total_pages,
                avg_chunk_size=row.avg_chunk_size,
                processing_time=row.processing_time,
                total_hits=row.total_hits,
                grounded_hits=row.grounded_hits,
                ungrounded_hits=row.ungrounded_hits,
            )

    async def save_tree_index(self, source_id: str, tree_index_json: str) -> None:
        await self.update_source(source_id, tree_index_json=tree_index_json)

    async def get_tree_index(self, source_id: str) -> str | None:
        async with self._session_factory() as session:
            row = await session.get(_SourceRow, source_id)
            if row is None:
                return None
            return row.tree_index_json

    async def get_tree_indexes(self, source_ids: list[str]) -> dict[str, str | None]:
        if not source_ids:
            return {}
        stmt = select(_SourceRow.id, _SourceRow.tree_index_json).where(
            _SourceRow.id.in_(source_ids)
        )
        async with self._session_factory() as session:
            rows = (await session.execute(stmt)).all()
        found = {row.id: row.tree_index_json for row in rows}
        # Preserve input order in output; missing source_ids map to None.
        return {sid: found.get(sid) for sid in source_ids}

    async def shutdown(self) -> None:
        await self._engine.dispose()

    @staticmethod
    def _row_to_source(row: _SourceRow) -> Source:
        return Source(
            source_id=row.id,
            status=row.status,
            metadata=json.loads(row.metadata_json),
            tags=json.loads(row.tags_json),
            chunk_count=row.chunk_count,
            embedding_model=row.embedding_model,
            file_hash=row.file_hash,
            created_at=row.created_at,
            stale=row.stale,
            knowledge_id=row.knowledge_id,
            source_type=row.source_type,
            source_weight=row.source_weight,
        )
