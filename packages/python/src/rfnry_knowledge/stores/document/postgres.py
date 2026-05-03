from typing import Any

from sqlalchemy import ColumnElement, String, Text, column, delete, func, literal, select, text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from rfnry_knowledge.common.logging import get_logger
from rfnry_knowledge.exceptions import ConfigurationError
from rfnry_knowledge.models import ContentMatch
from rfnry_knowledge.stores.document.excerpt import extract_window

logger = get_logger(__name__)


def _escape_like(text: str) -> str:
    """Escape LIKE/ILIKE special characters so they match literally."""
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


class _Base(DeclarativeBase):
    pass


class _SourceContentRow(_Base):
    __tablename__ = "knowledge_source_content"

    source_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    knowledge_id: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    source_type: Mapped[str | None] = mapped_column(String(100), nullable=True)
    title: Mapped[str | None] = mapped_column(String(500), nullable=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)


class PostgresDocumentStore:
    """Full-text document store backed by PostgreSQL (tsvector + ILIKE) or SQLite (LIKE fallback)."""

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
        headline_max_words: int = 200,
        headline_min_words: int = 80,
        headline_max_fragments: int = 3,
    ) -> None:
        if headline_max_words < 1:
            raise ConfigurationError("headline_max_words must be >= 1")
        if headline_min_words < 1:
            raise ConfigurationError("headline_min_words must be >= 1")
        if headline_max_fragments < 1:
            raise ConfigurationError("headline_max_fragments must be >= 1")
        if headline_min_words > headline_max_words:
            raise ConfigurationError(
                f"headline_min_words ({headline_min_words}) must be <= headline_max_words ({headline_max_words})"
            )
        if pool_timeout <= 0:
            raise ConfigurationError(f"pool_timeout must be > 0, got {pool_timeout}")
        # SQLAlchemy interprets -1 as "never recycle"; reject 0 or other negatives.
        if pool_recycle != -1 and pool_recycle <= 0:
            raise ConfigurationError(f"pool_recycle must be > 0 or -1 (disable), got {pool_recycle}")
        self._headline_max_words = headline_max_words
        self._headline_min_words = headline_min_words
        self._headline_max_fragments = headline_max_fragments
        parsed = make_url(url)
        if parsed.drivername == "postgresql":
            parsed = parsed.set(drivername="postgresql+asyncpg")
        elif parsed.drivername == "sqlite":
            parsed = parsed.set(drivername="sqlite+aiosqlite")

        kwargs: dict[str, Any] = {"echo": echo}
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
        self._is_postgres = parsed.drivername.startswith("postgresql")

    async def initialize(self) -> None:
        driver = self._engine.dialect.name
        if driver == "sqlite":
            logger.info(
                "document store initializing: driver=sqlite static pool (pool_size/max_overflow not applicable)"
            )
        else:
            pool = self._engine.sync_engine.pool
            pool_size = pool.size() if hasattr(pool, "size") else "n/a"
            overflow = getattr(pool, "_max_overflow", "n/a")
            recycle = getattr(pool, "_recycle", "n/a")
            timeout = getattr(pool, "_timeout", "n/a")
            logger.info(
                "document store initializing: driver=%s pool_size=%s max_overflow=%s pool_recycle=%ss pool_timeout=%ss",
                driver,
                pool_size,
                overflow,
                recycle,
                timeout,
            )
        async with self._engine.begin() as conn:
            await conn.run_sync(_Base.metadata.create_all)

            if self._is_postgres:
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))

                await conn.execute(
                    text(
                        "DO $$ BEGIN "
                        "ALTER TABLE knowledge_source_content "
                        "ADD COLUMN tsv tsvector GENERATED ALWAYS AS "
                        "(to_tsvector('english', coalesce(title, '') || ' ' || coalesce(content, ''))) STORED; "
                        "EXCEPTION WHEN duplicate_column THEN NULL; END $$;"
                    )
                )

                await conn.execute(
                    text(
                        "CREATE INDEX IF NOT EXISTS idx_knowledge_source_content_tsv ON knowledge_source_content USING GIN (tsv)"
                    )
                )

                await conn.execute(
                    text(
                        "CREATE INDEX IF NOT EXISTS idx_knowledge_source_content_trgm "
                        "ON knowledge_source_content USING GIN (content gin_trgm_ops)"
                    )
                )

        logger.info("document store tables initialized (postgres=%s)", self._is_postgres)

    async def store_content(
        self,
        source_id: str,
        knowledge_id: str | None,
        source_type: str | None,
        title: str,
        content: str,
    ) -> None:
        row = _SourceContentRow(
            source_id=source_id,
            knowledge_id=knowledge_id,
            source_type=source_type,
            title=title,
            content=content,
        )
        async with self._session_factory() as session:
            await session.merge(row)
            await session.commit()

    async def search_content(
        self,
        query: str,
        knowledge_id: str | None = None,
        source_type: str | None = None,
        top_k: int = 5,
    ) -> list[ContentMatch]:
        if not query or not query.strip():
            return []
        if self._is_postgres:
            return await self._search_postgres(query, knowledge_id, source_type, top_k)
        return await self._search_fallback(query, knowledge_id, source_type, top_k)

    async def get(self, source_id: str) -> str | None:
        """Return the stored full text for ``source_id``, or None if absent."""
        async with self._session_factory() as session:
            result = await session.execute(
                select(_SourceContentRow.content).where(_SourceContentRow.source_id == source_id)
            )
            row = result.scalar_one_or_none()
            return row

    async def delete_content(self, source_id: str) -> None:
        async with self._session_factory() as session:
            await session.execute(delete(_SourceContentRow).where(_SourceContentRow.source_id == source_id))
            await session.commit()

    async def shutdown(self) -> None:
        await self._engine.dispose()

    async def _search_postgres(
        self,
        query: str,
        knowledge_id: str | None,
        source_type: str | None,
        top_k: int,
    ) -> list[ContentMatch]:
        seen: dict[str, ContentMatch] = {}

        # Reference the generated tsvector column by name — it is not in the ORM
        # mapping because it is a GENERATED ALWAYS column added via DDL.
        tsv_col: ColumnElement[Any] = column("tsv")
        tsq = func.plainto_tsquery(literal("english"), literal(query))
        rank_col = func.ts_rank(tsv_col, tsq).label("rank")
        headline_opts = (
            f"MaxWords={self._headline_max_words},"
            f"MinWords={self._headline_min_words},"
            f"MaxFragments={self._headline_max_fragments}"
        )
        headline_col = func.ts_headline(
            literal("english"),
            _SourceContentRow.content,
            tsq,
            literal(headline_opts),
        ).label("headline")

        stmt = select(
            _SourceContentRow.source_id,
            _SourceContentRow.title,
            _SourceContentRow.source_type,
            rank_col,
            headline_col,
        ).where(tsv_col.op("@@")(tsq))
        if knowledge_id is not None:
            stmt = stmt.where(_SourceContentRow.knowledge_id == knowledge_id)
        if source_type is not None:
            stmt = stmt.where(_SourceContentRow.source_type == source_type)
        stmt = stmt.order_by(rank_col.desc()).limit(top_k)

        async with self._session_factory() as session:
            result = await session.execute(stmt)
            for row in result:
                seen[row.source_id] = ContentMatch(
                    source_id=row.source_id,
                    title=row.title,
                    excerpt=row.headline,
                    score=float(row.rank),
                    match_type="fulltext",
                    source_type=row.source_type,
                )

            if len(seen) < top_k:
                remaining = top_k - len(seen)
                ilike_stmt = select(
                    _SourceContentRow.source_id,
                    _SourceContentRow.title,
                    _SourceContentRow.content,
                    _SourceContentRow.source_type,
                ).where(_SourceContentRow.content.ilike(f"%{_escape_like(query)}%"))
                if knowledge_id is not None:
                    ilike_stmt = ilike_stmt.where(_SourceContentRow.knowledge_id == knowledge_id)
                if source_type is not None:
                    ilike_stmt = ilike_stmt.where(_SourceContentRow.source_type == source_type)
                ilike_stmt = ilike_stmt.limit(remaining + len(seen))

                result = await session.execute(ilike_stmt)
                for row in result:
                    if row.source_id not in seen:
                        seen[row.source_id] = ContentMatch(
                            source_id=row.source_id,
                            title=row.title,
                            excerpt=extract_window(row.content, query),
                            score=0.5,
                            match_type="exact",
                            source_type=row.source_type,
                        )
                        if len(seen) >= top_k:
                            break

        results = sorted(seen.values(), key=lambda m: m.score, reverse=True)
        return results[:top_k]

    async def _search_fallback(
        self,
        query: str,
        knowledge_id: str | None,
        source_type: str | None,
        top_k: int,
    ) -> list[ContentMatch]:
        stmt = select(_SourceContentRow).where(_SourceContentRow.content.ilike(f"%{_escape_like(query)}%"))
        if knowledge_id is not None:
            stmt = stmt.where(_SourceContentRow.knowledge_id == knowledge_id)
        if source_type is not None:
            stmt = stmt.where(_SourceContentRow.source_type == source_type)
        stmt = stmt.limit(top_k)

        async with self._session_factory() as session:
            result = await session.execute(stmt)
            rows = result.scalars().all()

        return [
            ContentMatch(
                source_id=row.source_id,
                title=row.title or "",
                excerpt=extract_window(row.content, query),
                score=0.5,
                match_type="exact",
                source_type=row.source_type,
            )
            for row in rows
        ]
