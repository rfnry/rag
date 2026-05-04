import json
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
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

from rfnry_knowledge.common.logging import get_logger
from rfnry_knowledge.exceptions import ConfigurationError, DuplicateSourceError, SourceNotFoundError
from rfnry_knowledge.models import Source, SourceStats
from rfnry_knowledge.telemetry.record import (
    IngestTelemetryRow,
    MemoryAddTelemetryRow,
    MemoryDeleteTelemetryRow,
    MemorySearchTelemetryRow,
    MemoryUpdateTelemetryRow,
    QueryTelemetryRow,
)

logger = get_logger(__name__)

# Current schema version. Bump on every additive migration so we can detect
# downgrade attempts and avoid double-applying ALTER statements under concurrent
# process start.
_SCHEMA_VERSION = 3


class _Base(DeclarativeBase):
    pass


class _SchemaMeta(_Base):
    __tablename__ = "knowledge_schema_meta"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[int] = mapped_column(Integer, nullable=False)


class _SourceRow(_Base):
    __tablename__ = "knowledge_sources"

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
    __tablename__ = "knowledge_source_stats"

    source_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("knowledge_sources.id", ondelete="CASCADE"), primary_key=True
    )
    total_chunks: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_pages: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    avg_chunk_size: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    processing_time: Mapped[float] = mapped_column(Float, nullable=False, default=0)
    total_hits: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    grounded_hits: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    ungrounded_hits: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class _PageAnalysisRow(_Base):
    __tablename__ = "knowledge_page_analyses"

    source_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    page_number: Mapped[int] = mapped_column(Integer, primary_key=True)
    page_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    data_json: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )

    __table_args__ = (
        Index("ix_page_analyses_source", "source_id"),
        ForeignKeyConstraint(
            ["source_id"],
            ["knowledge_sources.id"],
            ondelete="CASCADE",
        ),
    )


class _QueryTelemetryRow(_Base):
    __tablename__ = "knowledge_query_telemetry"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    schema_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    knowledge_id: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    query_id: Mapped[str] = mapped_column(String(64), nullable=False)
    mode: Mapped[str] = mapped_column(String(32), nullable=False)
    routing_decision: Mapped[str] = mapped_column(String(64), nullable=False)
    corpus_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    provider: Mapped[str | None] = mapped_column(String(64), nullable=True)
    model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    stop_reason: Mapped[str | None] = mapped_column(String(64), nullable=True)
    tokens_input: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    tokens_output: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    tokens_cache_creation: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    tokens_cache_read: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    llm_calls: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    duration_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    retrieval_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    grounding_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    generation_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    chunks_retrieved: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    methods_used_json: Mapped[str] = mapped_column(JSON, nullable=False, default="[]")
    method_durations_ms_json: Mapped[str] = mapped_column(JSON, nullable=False, default="{}")
    method_errors: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    grounding_decision: Mapped[str | None] = mapped_column(String(32), nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    outcome: Mapped[str] = mapped_column(String(32), nullable=False)
    error_type: Mapped[str | None] = mapped_column(String(128), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)


class _IngestTelemetryRow(_Base):
    __tablename__ = "knowledge_ingest_telemetry"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    schema_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    knowledge_id: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    source_id: Mapped[str] = mapped_column(String(64), nullable=False)
    ingest_id: Mapped[str] = mapped_column(String(64), nullable=False)
    source_type: Mapped[str | None] = mapped_column(String(100), nullable=True)
    chunks_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    pages_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    contextual_chunk_calls: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    contextual_chunk_skipped: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    document_expansion_calls: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    document_expansion_chunk_failures: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    vision_pages_analyzed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    vision_pages_skipped: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    graph_extraction_failed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    tokens_input: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    tokens_output: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    tokens_cache_creation: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    tokens_cache_read: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    llm_calls: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    duration_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    parse_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    chunk_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    enrichment_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    embed_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    persist_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    outcome: Mapped[str] = mapped_column(String(32), nullable=False)
    notes_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error_type: Mapped[str | None] = mapped_column(String(128), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)


class _MemoryAddTelemetryRow(_Base):
    __tablename__ = "knowledge_memory_add_telemetry"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    schema_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    memory_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    row_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    dropped_dedup_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    dropped_invalid_link_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    extraction_duration_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    semantic_duration_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    keyword_duration_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    entity_duration_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_duration_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    tokens_input: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    tokens_output: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    tokens_cache_creation: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    tokens_cache_read: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    llm_calls: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    outcome: Mapped[str] = mapped_column(String(32), nullable=False)
    error_type: Mapped[str | None] = mapped_column(String(128), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)


class _MemorySearchTelemetryRow(_Base):
    __tablename__ = "knowledge_memory_search_telemetry"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    schema_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    memory_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    result_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    top_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    methods_used_json: Mapped[str] = mapped_column(JSON, nullable=False, default="[]")
    method_durations_ms_json: Mapped[str] = mapped_column(JSON, nullable=False, default="{}")
    method_errors: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    duration_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    outcome: Mapped[str] = mapped_column(String(32), nullable=False)
    error_type: Mapped[str | None] = mapped_column(String(128), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)


class _MemoryUpdateTelemetryRow(_Base):
    __tablename__ = "knowledge_memory_update_telemetry"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    schema_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    memory_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    memory_row_id: Mapped[str] = mapped_column(String(64), nullable=False)
    text_before: Mapped[str | None] = mapped_column(Text, nullable=True)
    text_after: Mapped[str | None] = mapped_column(Text, nullable=True)
    entities_added: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    entities_removed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    duration_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    outcome: Mapped[str] = mapped_column(String(32), nullable=False)
    error_type: Mapped[str | None] = mapped_column(String(128), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)


class _MemoryDeleteTelemetryRow(_Base):
    __tablename__ = "knowledge_memory_delete_telemetry"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    schema_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    memory_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    memory_row_id: Mapped[str] = mapped_column(String(64), nullable=False)
    text_before: Mapped[str | None] = mapped_column(Text, nullable=True)
    duration_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    outcome: Mapped[str] = mapped_column(String(32), nullable=False)
    error_type: Mapped[str | None] = mapped_column(String(128), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)


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
            text("INSERT INTO knowledge_schema_meta (key, value) VALUES ('schema_version', 0) ON CONFLICT DO NOTHING")
            if conn.dialect.name == "postgresql"
            else text("INSERT OR IGNORE INTO knowledge_schema_meta (key, value) VALUES ('schema_version', 0)")
        )
        current = conn.execute(
            text("SELECT value FROM knowledge_schema_meta WHERE key = 'schema_version'")
        ).scalar_one()

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
                text("UPDATE knowledge_schema_meta SET value = :v WHERE key = 'schema_version'"),
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
        # Create knowledge_page_analyses if it doesn't exist yet.
        if not insp.has_table("knowledge_page_analyses"):
            _Base.metadata.create_all(
                bind=conn,
                tables=[_PageAnalysisRow.__table__],  # type: ignore[list-item]
            )
            logger.info("migrated table: knowledge_page_analyses")
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
        # Schema version 2: index on knowledge_sources.file_hash for find_by_hash performance.
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_knowledge_sources_file_hash ON knowledge_sources (file_hash)"))
        logger.info("ensured index: knowledge_sources.file_hash")

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
        stmt = select(_SourceRow.id, _SourceRow.tree_index_json).where(_SourceRow.id.in_(source_ids))
        async with self._session_factory() as session:
            rows = (await session.execute(stmt)).all()
        found = {row.id: row.tree_index_json for row in rows}
        # Preserve input order in output; missing source_ids map to None.
        return {sid: found.get(sid) for sid in source_ids}

    async def upsert_page_analyses(
        self,
        source_id: str,
        analyses: list[dict],
    ) -> None:
        """Upsert per-page analyses keyed by (source_id, page_number).

        ``analyses`` is a list of ``{"page_number": int, "data": dict}`` entries.
        If ``data`` contains a ``page_hash`` key, it is hoisted into the indexed
        ``page_hash`` column for fast lookup; otherwise the column stays NULL.
        """
        if not analyses:
            return
        async with self._session_factory() as session:
            dialect = self._engine.dialect.name
            for a in analyses:
                data = a["data"]
                values = dict(
                    source_id=source_id,
                    page_number=a["page_number"],
                    page_hash=data.get("page_hash"),
                    data_json=json.dumps(data),
                    updated_at=datetime.now(UTC),
                )
                upsert_stmt: Any
                if dialect == "sqlite":
                    from sqlalchemy.dialects.sqlite import insert as sqlite_insert

                    upsert_stmt = sqlite_insert(_PageAnalysisRow).values(**values)
                    upsert_stmt = upsert_stmt.on_conflict_do_update(
                        index_elements=["source_id", "page_number"],
                        set_={
                            "page_hash": upsert_stmt.excluded.page_hash,
                            "data_json": upsert_stmt.excluded.data_json,
                            "updated_at": upsert_stmt.excluded.updated_at,
                        },
                    )
                elif dialect == "postgresql":
                    from sqlalchemy.dialects.postgresql import insert as pg_insert

                    upsert_stmt = pg_insert(_PageAnalysisRow).values(**values)
                    upsert_stmt = upsert_stmt.on_conflict_do_update(
                        index_elements=["source_id", "page_number"],
                        set_={
                            "page_hash": upsert_stmt.excluded.page_hash,
                            "data_json": upsert_stmt.excluded.data_json,
                            "updated_at": upsert_stmt.excluded.updated_at,
                        },
                    )
                else:
                    raise NotImplementedError(f"upsert_page_analyses not implemented for dialect {dialect!r}")
                await session.execute(upsert_stmt)
            await session.commit()

    async def get_page_analyses(self, source_id: str) -> list[dict]:
        """Return all page analyses for a source, ordered by page_number.

        Returns a list of {"page_number": int, "page_hash": str | None, "data": dict}.
        """
        async with self._session_factory() as session:
            stmt = (
                select(_PageAnalysisRow)
                .where(_PageAnalysisRow.source_id == source_id)
                .order_by(_PageAnalysisRow.page_number)
            )
            rows = (await session.execute(stmt)).scalars().all()
        return [
            {
                "page_number": r.page_number,
                "page_hash": r.page_hash,
                "data": json.loads(r.data_json),
            }
            for r in rows
        ]

    async def get_page_analyses_by_hash(
        self,
        page_hashes: list[str],
        knowledge_id: str | None,
    ) -> dict[str, dict]:
        """Return {page_hash: data_dict} for any previously-analyzed page whose hash matches.

        When ``knowledge_id`` is provided, restrict to sources from the same knowledge
        (so per-knowledge prompt variations don't collide). When None, allow any match.
        Rows whose ``page_hash`` column is NULL or empty are never returned.
        """
        if not page_hashes:
            return {}
        non_empty = [h for h in page_hashes if h]
        if not non_empty:
            return {}

        from sqlalchemy import and_

        stmt = select(_PageAnalysisRow).where(_PageAnalysisRow.page_hash.in_(non_empty))
        if knowledge_id is not None:
            stmt = stmt.join(
                _SourceRow,
                and_(
                    _SourceRow.id == _PageAnalysisRow.source_id,
                    _SourceRow.knowledge_id == knowledge_id,
                ),
            )
        async with self._session_factory() as session:
            rows = (await session.execute(stmt)).scalars().all()

        # If multiple rows share the same hash (same page in multiple sources),
        # last one wins — fine for cache purposes.
        return {r.page_hash: json.loads(r.data_json) for r in rows if r.page_hash}

    async def get_page_analysis(
        self,
        source_id: str,
        page_number: int,
    ) -> dict | None:
        """Return the single page analysis's data dict for (source_id, page_number), or None."""
        async with self._session_factory() as session:
            stmt = select(_PageAnalysisRow).where(
                _PageAnalysisRow.source_id == source_id,
                _PageAnalysisRow.page_number == page_number,
            )
            row = (await session.execute(stmt)).scalar_one_or_none()
        if row is None:
            return None
        return json.loads(row.data_json)

    async def insert_query_telemetry(self, row: QueryTelemetryRow) -> None:
        orm_row = _QueryTelemetryRow(
            schema_version=row.schema_version,
            at=row.at,
            knowledge_id=row.knowledge_id,
            query_id=row.query_id,
            mode=row.mode,
            routing_decision=row.routing_decision,
            corpus_tokens=row.corpus_tokens,
            provider=row.provider,
            model=row.model,
            stop_reason=row.stop_reason,
            tokens_input=row.tokens_input,
            tokens_output=row.tokens_output,
            tokens_cache_creation=row.tokens_cache_creation,
            tokens_cache_read=row.tokens_cache_read,
            llm_calls=row.llm_calls,
            duration_ms=row.duration_ms,
            retrieval_ms=row.retrieval_ms,
            grounding_ms=row.grounding_ms,
            generation_ms=row.generation_ms,
            chunks_retrieved=row.chunks_retrieved,
            methods_used_json=json.dumps(row.methods_used),
            method_durations_ms_json=json.dumps(row.method_durations_ms),
            method_errors=row.method_errors,
            grounding_decision=row.grounding_decision,
            confidence=row.confidence,
            outcome=row.outcome,
            error_type=row.error_type,
            error_message=row.error_message,
        )
        async with self._session_factory() as session:
            session.add(orm_row)
            await session.commit()

    async def insert_ingest_telemetry(self, row: IngestTelemetryRow) -> None:
        orm_row = _IngestTelemetryRow(
            schema_version=row.schema_version,
            at=row.at,
            knowledge_id=row.knowledge_id,
            source_id=row.source_id,
            ingest_id=row.ingest_id,
            source_type=row.source_type,
            chunks_count=row.chunks_count,
            pages_count=row.pages_count,
            contextual_chunk_calls=row.contextual_chunk_calls,
            contextual_chunk_skipped=row.contextual_chunk_skipped,
            document_expansion_calls=row.document_expansion_calls,
            document_expansion_chunk_failures=row.document_expansion_chunk_failures,
            vision_pages_analyzed=row.vision_pages_analyzed,
            vision_pages_skipped=row.vision_pages_skipped,
            graph_extraction_failed=row.graph_extraction_failed,
            tokens_input=row.tokens_input,
            tokens_output=row.tokens_output,
            tokens_cache_creation=row.tokens_cache_creation,
            tokens_cache_read=row.tokens_cache_read,
            llm_calls=row.llm_calls,
            duration_ms=row.duration_ms,
            parse_ms=row.parse_ms,
            chunk_ms=row.chunk_ms,
            enrichment_ms=row.enrichment_ms,
            embed_ms=row.embed_ms,
            persist_ms=row.persist_ms,
            outcome=row.outcome,
            notes_count=row.notes_count,
            error_type=row.error_type,
            error_message=row.error_message,
        )
        async with self._session_factory() as session:
            session.add(orm_row)
            await session.commit()

    async def insert_memory_add_telemetry(self, row: MemoryAddTelemetryRow) -> None:
        orm_row = _MemoryAddTelemetryRow(
            schema_version=row.schema_version,
            at=row.at,
            memory_id=row.memory_id,
            row_count=row.row_count,
            dropped_dedup_count=row.dropped_dedup_count,
            dropped_invalid_link_count=row.dropped_invalid_link_count,
            extraction_duration_ms=row.extraction_duration_ms,
            semantic_duration_ms=row.semantic_duration_ms,
            keyword_duration_ms=row.keyword_duration_ms,
            entity_duration_ms=row.entity_duration_ms,
            total_duration_ms=row.total_duration_ms,
            tokens_input=row.tokens_input,
            tokens_output=row.tokens_output,
            tokens_cache_creation=row.tokens_cache_creation,
            tokens_cache_read=row.tokens_cache_read,
            llm_calls=row.llm_calls,
            outcome=row.outcome,
            error_type=row.error_type,
            error_message=row.error_message,
        )
        async with self._session_factory() as session:
            session.add(orm_row)
            await session.commit()

    async def insert_memory_search_telemetry(self, row: MemorySearchTelemetryRow) -> None:
        orm_row = _MemorySearchTelemetryRow(
            schema_version=row.schema_version,
            at=row.at,
            memory_id=row.memory_id,
            result_count=row.result_count,
            top_score=row.top_score,
            methods_used_json=json.dumps(row.methods_used),
            method_durations_ms_json=json.dumps(row.method_durations_ms),
            method_errors=row.method_errors,
            duration_ms=row.duration_ms,
            outcome=row.outcome,
            error_type=row.error_type,
            error_message=row.error_message,
        )
        async with self._session_factory() as session:
            session.add(orm_row)
            await session.commit()

    async def insert_memory_update_telemetry(self, row: MemoryUpdateTelemetryRow) -> None:
        orm_row = _MemoryUpdateTelemetryRow(
            schema_version=row.schema_version,
            at=row.at,
            memory_id=row.memory_id,
            memory_row_id=row.memory_row_id,
            text_before=row.text_before,
            text_after=row.text_after,
            entities_added=row.entities_added,
            entities_removed=row.entities_removed,
            duration_ms=row.duration_ms,
            outcome=row.outcome,
            error_type=row.error_type,
            error_message=row.error_message,
        )
        async with self._session_factory() as session:
            session.add(orm_row)
            await session.commit()

    async def insert_memory_delete_telemetry(self, row: MemoryDeleteTelemetryRow) -> None:
        orm_row = _MemoryDeleteTelemetryRow(
            schema_version=row.schema_version,
            at=row.at,
            memory_id=row.memory_id,
            memory_row_id=row.memory_row_id,
            text_before=row.text_before,
            duration_ms=row.duration_ms,
            outcome=row.outcome,
            error_type=row.error_type,
            error_message=row.error_message,
        )
        async with self._session_factory() as session:
            session.add(orm_row)
            await session.commit()

    async def list_query_telemetry(
        self,
        *,
        knowledge_id: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 1000,
    ) -> list[QueryTelemetryRow]:
        stmt = select(_QueryTelemetryRow)
        if knowledge_id is not None:
            stmt = stmt.where(_QueryTelemetryRow.knowledge_id == knowledge_id)
        if since is not None:
            stmt = stmt.where(_QueryTelemetryRow.at >= since)
        if until is not None:
            stmt = stmt.where(_QueryTelemetryRow.at <= until)
        stmt = stmt.order_by(_QueryTelemetryRow.at.desc()).limit(limit)
        async with self._session_factory() as session:
            rows = (await session.execute(stmt)).scalars().all()
        return [self._query_orm_to_record(r) for r in rows]

    async def list_ingest_telemetry(
        self,
        *,
        knowledge_id: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 1000,
    ) -> list[IngestTelemetryRow]:
        stmt = select(_IngestTelemetryRow)
        if knowledge_id is not None:
            stmt = stmt.where(_IngestTelemetryRow.knowledge_id == knowledge_id)
        if since is not None:
            stmt = stmt.where(_IngestTelemetryRow.at >= since)
        if until is not None:
            stmt = stmt.where(_IngestTelemetryRow.at <= until)
        stmt = stmt.order_by(_IngestTelemetryRow.at.desc()).limit(limit)
        async with self._session_factory() as session:
            rows = (await session.execute(stmt)).scalars().all()
        return [self._ingest_orm_to_record(r) for r in rows]

    @staticmethod
    def _query_orm_to_record(r: _QueryTelemetryRow) -> QueryTelemetryRow:
        methods_used_raw = r.methods_used_json
        method_durations_raw = r.method_durations_ms_json
        methods_used = json.loads(methods_used_raw) if isinstance(methods_used_raw, str) else (methods_used_raw or [])
        method_durations = (
            json.loads(method_durations_raw) if isinstance(method_durations_raw, str) else (method_durations_raw or {})
        )
        return QueryTelemetryRow(
            schema_version=r.schema_version,
            at=r.at,
            knowledge_id=r.knowledge_id,
            query_id=r.query_id,
            mode=r.mode,  # type: ignore[arg-type]
            routing_decision=r.routing_decision,
            corpus_tokens=r.corpus_tokens,
            provider=r.provider,
            model=r.model,
            stop_reason=r.stop_reason,
            tokens_input=r.tokens_input,
            tokens_output=r.tokens_output,
            tokens_cache_creation=r.tokens_cache_creation,
            tokens_cache_read=r.tokens_cache_read,
            llm_calls=r.llm_calls,
            duration_ms=r.duration_ms,
            retrieval_ms=r.retrieval_ms,
            grounding_ms=r.grounding_ms,
            generation_ms=r.generation_ms,
            chunks_retrieved=r.chunks_retrieved,
            methods_used=methods_used,
            method_durations_ms=method_durations,
            method_errors=r.method_errors,
            grounding_decision=r.grounding_decision,  # type: ignore[arg-type]
            confidence=r.confidence,
            outcome=r.outcome,  # type: ignore[arg-type]
            error_type=r.error_type,
            error_message=r.error_message,
        )

    @staticmethod
    def _ingest_orm_to_record(r: _IngestTelemetryRow) -> IngestTelemetryRow:
        return IngestTelemetryRow(
            schema_version=r.schema_version,
            at=r.at,
            knowledge_id=r.knowledge_id,
            source_id=r.source_id,
            ingest_id=r.ingest_id,
            source_type=r.source_type,
            chunks_count=r.chunks_count,
            pages_count=r.pages_count,
            contextual_chunk_calls=r.contextual_chunk_calls,
            contextual_chunk_skipped=r.contextual_chunk_skipped,
            document_expansion_calls=r.document_expansion_calls,
            document_expansion_chunk_failures=r.document_expansion_chunk_failures,
            vision_pages_analyzed=r.vision_pages_analyzed,
            vision_pages_skipped=r.vision_pages_skipped,
            graph_extraction_failed=r.graph_extraction_failed,
            tokens_input=r.tokens_input,
            tokens_output=r.tokens_output,
            tokens_cache_creation=r.tokens_cache_creation,
            tokens_cache_read=r.tokens_cache_read,
            llm_calls=r.llm_calls,
            duration_ms=r.duration_ms,
            parse_ms=r.parse_ms,
            chunk_ms=r.chunk_ms,
            enrichment_ms=r.enrichment_ms,
            embed_ms=r.embed_ms,
            persist_ms=r.persist_ms,
            outcome=r.outcome,  # type: ignore[arg-type]
            notes_count=r.notes_count,
            error_type=r.error_type,
            error_message=r.error_message,
        )

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
