import pytest
from sqlalchemy import Column, Integer, MetaData, String, Table, inspect, text
from sqlalchemy.dialects import sqlite
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.schema import ColumnDefault

from rfnry_rag.retrieval.stores.metadata.sqlalchemy import SQLAlchemyMetadataStore


def test_render_default_literal_quotes_single_quote_string() -> None:
    dialect = sqlite.dialect()
    rendered = SQLAlchemyMetadataStore._render_default_literal("O'Brien", dialect)
    assert rendered == "'O''Brien'"


@pytest.mark.parametrize(
    "value, expected",
    [
        (True, "1"),
        (False, "0"),
        (None, "NULL"),
        (42, "42"),
        (3.14, "3.14"),
        ("hello", "'hello'"),
    ],
)
def test_render_default_literal_handles_basic_types(value, expected) -> None:
    dialect = sqlite.dialect()
    assert SQLAlchemyMetadataStore._render_default_literal(value, dialect) == expected


@pytest.mark.asyncio
async def test_migrate_adds_column_with_quote_containing_default(tmp_path) -> None:
    db_path = tmp_path / "mig.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")

    # Seed: create an empty table the migration will add a column to
    async with engine.begin() as conn:
        await conn.execute(text("CREATE TABLE widgets (id INTEGER PRIMARY KEY)"))

    # Build a fake metadata containing a target column with a tricky string default
    meta = MetaData()
    Table(
        "widgets",
        meta,
        Column("id", Integer, primary_key=True),
        Column("note", String, nullable=False, default="O'Brien"),
    )

    async with engine.begin() as conn:

        def migrate(sync_conn):
            insp = inspect(sync_conn)
            table = meta.tables["widgets"]
            existing = {c["name"] for c in insp.get_columns("widgets")}
            for column in table.columns:
                if column.name in existing:
                    continue
                col_type = column.type.compile(sync_conn.dialect)
                default = ""
                if column.default is not None and isinstance(column.default, ColumnDefault):
                    literal = SQLAlchemyMetadataStore._render_default_literal(column.default.arg, sync_conn.dialect)
                    default = f" DEFAULT {literal}"
                nullable = "" if column.nullable else " NOT NULL"
                sync_conn.execute(
                    text(f'ALTER TABLE "widgets" ADD COLUMN "{column.name}" {col_type}{nullable}{default}')
                )

        await conn.run_sync(migrate)

    async with engine.begin() as conn:
        cols = await conn.run_sync(lambda c: inspect(c).get_columns("widgets"))
        names = {c["name"] for c in cols}
        assert "note" in names

        await conn.execute(text("INSERT INTO widgets (id) VALUES (1)"))
        result = await conn.execute(text("SELECT note FROM widgets WHERE id = 1"))
        row = result.fetchone()
        assert row is not None
        assert row[0] == "O'Brien"

    await engine.dispose()


def test_migrate_source_does_not_contain_unquoted_fstring_default() -> None:
    """The raw `f" DEFAULT '{val}'"` pattern in _migrate_missing_columns is unsafe
    for values containing single quotes. This test guards the fix — the file must
    not contain that raw pattern anymore."""
    from pathlib import Path

    src = Path("src/rfnry_rag/retrieval/stores/metadata/sqlalchemy.py").read_text()
    # The old, unsafe pattern used bare interpolation inside literal quotes
    assert "f\" DEFAULT '{val}'\"" not in src
    assert "f\" DEFAULT '{val}'\"" not in src


@pytest.mark.asyncio
async def test_schema_version_table_populated_on_initialize(tmp_path) -> None:
    """After initialize, rag_schema_meta contains the current version."""
    from rfnry_rag.retrieval.stores.metadata.sqlalchemy import _SCHEMA_VERSION

    store = SQLAlchemyMetadataStore(f"sqlite:///{tmp_path}/m.db")
    await store.initialize()

    async with store._engine.connect() as conn:
        result = await conn.execute(text("SELECT value FROM rag_schema_meta WHERE key = 'schema_version'"))
        row = result.fetchone()
        assert row is not None
        assert row[0] == _SCHEMA_VERSION

    await store.shutdown()


@pytest.mark.asyncio
async def test_schema_version_refuses_downgrade(tmp_path) -> None:
    """If the DB has a higher schema_version than code, initialize() must refuse."""
    from rfnry_rag.retrieval.stores.metadata.sqlalchemy import _SCHEMA_VERSION

    store = SQLAlchemyMetadataStore(f"sqlite:///{tmp_path}/m.db")
    await store.initialize()
    # Poison the version row with a future value
    async with store._engine.begin() as conn:
        await conn.execute(
            text("UPDATE rag_schema_meta SET value = :v WHERE key = 'schema_version'"),
            {"v": _SCHEMA_VERSION + 99},
        )
    await store.shutdown()

    store2 = SQLAlchemyMetadataStore(f"sqlite:///{tmp_path}/m.db")
    with pytest.raises(RuntimeError, match="downgrade is not supported"):
        await store2.initialize()
    await store2.shutdown()


@pytest.mark.asyncio
async def test_repeated_initialize_is_idempotent(tmp_path) -> None:
    """initialize() is safe to call repeatedly (e.g. fresh process starts against
    an already-migrated DB)."""
    url = f"sqlite:///{tmp_path}/m.db"

    for _ in range(3):
        store = SQLAlchemyMetadataStore(url)
        await store.initialize()
        await store.shutdown()
