from typing import Any

from sqlalchemy import String, Text, delete, select, text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from rfnry_rag.retrieval.common.logging import get_logger
from rfnry_rag.retrieval.common.models import ContentMatch
from rfnry_rag.retrieval.stores.document.excerpt import extract_window

logger = get_logger(__name__)


def _escape_like(text: str) -> str:
    """Escape LIKE/ILIKE special characters so they match literally."""
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


class _Base(DeclarativeBase):
    pass


class _SourceContentRow(_Base):
    __tablename__ = "rag_source_content"

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
    ) -> None:
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
        async with self._engine.begin() as conn:
            await conn.run_sync(_Base.metadata.create_all)

            if self._is_postgres:
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))

                await conn.execute(
                    text(
                        "DO $$ BEGIN "
                        "ALTER TABLE rag_source_content "
                        "ADD COLUMN tsv tsvector GENERATED ALWAYS AS "
                        "(to_tsvector('english', coalesce(title, '') || ' ' || coalesce(content, ''))) STORED; "
                        "EXCEPTION WHEN duplicate_column THEN NULL; END $$;"
                    )
                )

                await conn.execute(
                    text("CREATE INDEX IF NOT EXISTS idx_rag_source_content_tsv ON rag_source_content USING GIN (tsv)")
                )

                await conn.execute(
                    text(
                        "CREATE INDEX IF NOT EXISTS idx_rag_source_content_trgm "
                        "ON rag_source_content USING GIN (content gin_trgm_ops)"
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

        where_clauses = ["tsv @@ plainto_tsquery('english', :query)"]
        params: dict[str, Any] = {"query": query}
        if knowledge_id is not None:
            where_clauses.append("knowledge_id = :knowledge_id")
            params["knowledge_id"] = knowledge_id
        if source_type is not None:
            where_clauses.append("source_type = :source_type")
            params["source_type"] = source_type

        where_sql = " AND ".join(where_clauses)
        tsquery_sql = text(
            f"SELECT source_id, title, source_type, "  # noqa: S608
            f"ts_rank(tsv, plainto_tsquery('english', :query)) AS rank, "
            f"ts_headline('english', content, plainto_tsquery('english', :query), "
            f"'MaxWords=200,MinWords=80,MaxFragments=3') AS headline "
            f"FROM rag_source_content WHERE {where_sql} "
            f"ORDER BY rank DESC LIMIT :top_k"
        )
        params["top_k"] = top_k

        async with self._session_factory() as session:
            result = await session.execute(tsquery_sql, params)
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
                ilike_where = ["content ILIKE :pattern"]
                ilike_params: dict[str, Any] = {"pattern": f"%{_escape_like(query)}%"}
                if knowledge_id is not None:
                    ilike_where.append("knowledge_id = :knowledge_id")
                    ilike_params["knowledge_id"] = knowledge_id
                if source_type is not None:
                    ilike_where.append("source_type = :source_type")
                    ilike_params["source_type"] = source_type

                ilike_where_sql = " AND ".join(ilike_where)
                ilike_sql = text(
                    f"SELECT source_id, title, content, source_type "  # noqa: S608
                    f"FROM rag_source_content WHERE {ilike_where_sql} "
                    f"LIMIT :limit"
                )
                ilike_params["limit"] = remaining + len(seen)

                result = await session.execute(ilike_sql, ilike_params)
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
