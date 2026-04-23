# Retrieval SDK Audit Fixes — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task.

**Goal:** Address all 14 findings from the 2026-04-23 review of `packages/python/src/rfnry_rag/retrieval/` — 2 credential leaks, 2 orchestration correctness bugs, 3 modularity gaps, and 7 production-hardening items.

**Architecture:** Each task is self-contained and TDD-driven. Phases are ordered by risk: correctness/security → modularity → hardening → nice-to-haves. Most tasks touch 1–2 files and ship as an independent commit so the branch can be bisected cleanly.

**Tech Stack:** Python 3.12, pytest (asyncio_mode=auto), Ruff, MyPy, SQLAlchemy async, asyncpg/aiosqlite, Qdrant, Neo4j, BAML. Tasks run via `poe test`, `poe check`, `poe typecheck`.

**Conventions:**
- All paths relative to `packages/python/` unless absolute
- Test files live under `src/rfnry_rag/retrieval/tests/`
- Every task ends with `poe check` + `poe typecheck` + `poe test` green before commit
- Commit messages use conventional commits (see `git log --oneline -20` for existing style)

---

## Phase 1 — Must-fix (correctness + security)

### Task 1: Stop leaking DB credentials via `render_as_string(hide_password=False)`

**Finding C1.** Passing the rendered URL string to `create_async_engine` materializes plaintext credentials into a local variable where they can leak through logs, tracebacks, and engine `repr()`. SQLAlchemy's `URL` object is a first-class input to `create_async_engine` and never serializes credentials unless asked.

**Files:**
- Modify: `src/rfnry_rag/retrieval/stores/metadata/sqlalchemy.py:74-76`
- Modify: `src/rfnry_rag/retrieval/stores/document/postgres.py:43-45`
- Test: `src/rfnry_rag/retrieval/tests/test_store_credential_safety.py` (new)

**Step 1: Write the failing test**

```python
# src/rfnry_rag/retrieval/tests/test_store_credential_safety.py
import pytest

from rfnry_rag.retrieval.stores.document.postgres import PostgresDocumentStore
from rfnry_rag.retrieval.stores.metadata.sqlalchemy import SQLAlchemyMetadataStore


@pytest.mark.parametrize(
    "store_cls, url",
    [
        (SQLAlchemyMetadataStore, "postgresql://alice:s3cr3t@localhost:5432/db"),
        (PostgresDocumentStore, "postgresql://alice:s3cr3t@localhost:5432/db"),
    ],
)
def test_store_does_not_materialize_password(store_cls, url):
    store = store_cls(url)
    # password must never be reachable via repr or engine.url.render_as_string (default hides it)
    rendered = store._engine.url.render_as_string()
    assert "s3cr3t" not in rendered
    assert "s3cr3t" not in repr(store._engine)
    # the URL object itself still carries the password (for the driver)
    assert store._engine.url.password == "s3cr3t"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest src/rfnry_rag/retrieval/tests/test_store_credential_safety.py -v`
Expected: PASS actually — SQLAlchemy's default `render_as_string()` already hides the password. The real risk is *our code* calling `hide_password=False`. Replace the test with one that targets the smell directly:

```python
import ast
from pathlib import Path

TARGETS = [
    "src/rfnry_rag/retrieval/stores/metadata/sqlalchemy.py",
    "src/rfnry_rag/retrieval/stores/document/postgres.py",
]

def test_stores_do_not_call_hide_password_false():
    for rel in TARGETS:
        src = Path(rel).read_text()
        assert "hide_password=False" not in src, (
            f"{rel} must not render URLs with hide_password=False — "
            "pass the URL object directly to create_async_engine instead"
        )
```

Re-run: Expected FAIL — both files currently contain `hide_password=False`.

**Step 3: Fix `sqlalchemy.py`**

```python
# src/rfnry_rag/retrieval/stores/metadata/sqlalchemy.py
# Replace lines 69-77 with:
parsed = make_url(url)
if parsed.drivername == "postgresql":
    parsed = parsed.set(drivername="postgresql+asyncpg")
elif parsed.drivername == "sqlite":
    parsed = parsed.set(drivername="sqlite+aiosqlite")

self._engine = create_async_engine(parsed, echo=False)
self._session_factory = async_sessionmaker(self._engine, class_=AsyncSession, expire_on_commit=False)
```

**Step 4: Fix `postgres.py` identically**

Same shape — drop the `connection_string` local; pass `parsed` to `create_async_engine`.

**Step 5: Run tests + lint + typecheck**

```bash
uv run pytest src/rfnry_rag/retrieval/tests/test_store_credential_safety.py src/rfnry_rag/retrieval/tests/test_postgres_document_store.py -v
uv run poe check && uv run poe typecheck
```

Expected: all pass.

**Step 6: Commit**

```bash
git add src/rfnry_rag/retrieval/stores/metadata/sqlalchemy.py \
        src/rfnry_rag/retrieval/stores/document/postgres.py \
        src/rfnry_rag/retrieval/tests/test_store_credential_safety.py
git commit -m "fix: stop materializing DB credentials via render_as_string(hide_password=False)"
```

---

### Task 2: Replace f-string DDL in `_migrate_missing_columns` with safe quoting

**Finding C2.** `sqlalchemy.py:106` builds `ALTER TABLE` DDL via f-string with uncontrolled `default` values. Any string default containing `'` will produce broken SQL today.

**Files:**
- Modify: `src/rfnry_rag/retrieval/stores/metadata/sqlalchemy.py:85-108`
- Test: `src/rfnry_rag/retrieval/tests/test_metadata_store_migration.py` (new)

**Step 1: Write failing test**

```python
# src/rfnry_rag/retrieval/tests/test_metadata_store_migration.py
import pytest
from sqlalchemy import Column, Integer, MetaData, String, Table, inspect, text
from sqlalchemy.ext.asyncio import create_async_engine

from rfnry_rag.retrieval.stores.metadata.sqlalchemy import SQLAlchemyMetadataStore


@pytest.mark.asyncio
async def test_migrate_survives_string_default_with_single_quote(tmp_path):
    db = tmp_path / "t.db"
    store = SQLAlchemyMetadataStore(f"sqlite:///{db}")
    await store.initialize()

    # Simulate a model that declares a column with a quote-containing default
    meta = MetaData()
    tbl = Table(
        "rag_source",
        meta,
        Column("id", String, primary_key=True),
        Column("note", String, nullable=False, default="O'Brien"),
        extend_existing=True,
    )
    # Run the migration logic on our fake metadata
    async with store._engine.begin() as conn:
        def migrate(sync_conn):
            insp = inspect(sync_conn)
            existing = {c["name"] for c in insp.get_columns("rag_source")}
            for col in tbl.columns:
                if col.name not in existing and col.name != "id":
                    SQLAlchemyMetadataStore._migrate_missing_columns_for_table(sync_conn, tbl)
                    break
        await conn.run_sync(migrate)

    async with store._engine.connect() as conn:
        insp = await conn.run_sync(lambda c: inspect(c).get_columns("rag_source"))
    assert any(c["name"] == "note" for c in insp)
```

Also add a unit test for the quoting helper directly once Step 3 adds it.

**Step 2: Run — expect FAIL** (helper not yet extracted; current implementation raises `OperationalError` on the bad quote).

**Step 3: Extract a safe helper**

In `sqlalchemy.py`, replace `_migrate_missing_columns` with a version that uses `sqlalchemy.schema.DDL` + `sqlalchemy.sql.quoted_name` and a safe literal renderer:

```python
from sqlalchemy import DDL
from sqlalchemy.sql import quoted_name


@staticmethod
def _render_default_literal(val: object, dialect) -> str:
    """Render a Python default value as a SQL literal, dialect-safe."""
    if isinstance(val, bool):
        return "1" if val else "0"
    if val is None:
        return "NULL"
    if isinstance(val, (int, float)):
        return str(val)
    # Strings and everything else: use the dialect's literal processor
    from sqlalchemy import String
    proc = String().literal_processor(dialect=dialect)
    return proc(str(val))


@staticmethod
def _migrate_missing_columns(conn) -> None:
    insp = inspect(conn)
    for table in _Base.metadata.sorted_tables:
        if not insp.has_table(table.name):
            continue
        existing = {col["name"] for col in insp.get_columns(table.name)}
        for column in table.columns:
            if column.name in existing:
                continue
            col_type = column.type.compile(conn.dialect)
            nullable = "" if column.nullable else " NOT NULL"
            default = ""
            if column.default is not None and isinstance(column.default, ColumnDefault):
                literal = SQLAlchemyMetadataStore._render_default_literal(
                    column.default.arg, conn.dialect
                )
                default = f" DEFAULT {literal}"
            safe_table = quoted_name(table.name, quote=True)
            safe_col = quoted_name(column.name, quote=True)
            conn.execute(DDL(
                f"ALTER TABLE {safe_table} ADD COLUMN {safe_col} {col_type}{nullable}{default}"
            ))
            logger.info("migrated column: %s.%s", table.name, column.name)
```

The `literal_processor` does dialect-correct quoting — `O'Brien` becomes `'O''Brien'` on PostgreSQL and SQLite.

**Step 4: Re-run tests** — expect PASS.

**Step 5: Commit**

```bash
git add src/rfnry_rag/retrieval/stores/metadata/sqlalchemy.py \
        src/rfnry_rag/retrieval/tests/test_metadata_store_migration.py
git commit -m "fix: use dialect-safe literal quoting in metadata column migration"
```

---

### Task 3: Fix tree-search result fusion bug in `_retrieve_chunks`

**Finding from HIGH correctness set.** `server.py:858-863` calls `unstructured.retrieve(..., tree_chunks=tree_chunks)` a *second* time when tree chunks exist, overwriting the already-merged `chunks` variable and losing the first retrieval's work.

**Files:**
- Modify: `src/rfnry_rag/retrieval/server.py:838-867`
- Test: `src/rfnry_rag/retrieval/tests/test_tree_search_fusion.py` (new)

**Step 1: Read the current implementation**

Read `server.py:838-867` and the adjacent `_run_tree_search` to understand the shape. The fix is: run tree search *alongside* unstructured/structured retrieval, then fuse via the existing `tree_chunks` parameter of `RetrievalService.retrieve()` (which already handles the tree-chunks fusion at `modules/retrieval/search/service.py:71-74`). The second `unstructured.retrieve()` call is redundant.

**Step 2: Write failing test**

```python
# src/rfnry_rag/retrieval/tests/test_tree_search_fusion.py
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from rfnry_rag.retrieval.common.models import RetrievedChunk
from rfnry_rag.retrieval.server import RagEngine


def _chunk(cid: str, score: float = 1.0) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=cid, content=cid, score=score, source_id="s", page_number=1)


@pytest.mark.asyncio
async def test_tree_search_fuses_with_unstructured_not_overwrite():
    engine = RagEngine.__new__(RagEngine)
    engine._config = SimpleNamespace(persistence=SimpleNamespace(metadata_store=object()))

    unstructured = SimpleNamespace(retrieve=AsyncMock(return_value=[_chunk("u1"), _chunk("u2")]))
    engine._get_retrieval = lambda _c: (unstructured, None)
    engine._build_retrieval_query = lambda text, history: text
    engine._run_tree_search = AsyncMock(return_value=[_chunk("t1")])
    engine._tree_search_service = object()

    chunks = await engine._retrieve_chunks(
        collection="default", text="q", history=None, knowledge_id=None, min_score=None
    )
    # unstructured.retrieve must be called exactly once with tree_chunks=[...]
    assert unstructured.retrieve.call_count == 1
    call = unstructured.retrieve.await_args
    assert call.kwargs.get("tree_chunks") == [_chunk("t1")]
    ids = {c.chunk_id for c in chunks}
    assert ids >= {"u1", "u2"}  # unstructured results preserved
```

**Step 3: Run — expect FAIL** (call count will be 2).

**Step 4: Implement the fix**

Replace `server.py:838-867` (the `_retrieve_chunks` body after `retrieval_query = ...`):

```python
# Run unstructured + structured + tree search concurrently
tree_chunks: list[RetrievedChunk] = []
if self._tree_search_service and self._config.persistence.metadata_store:
    tree_chunks = await self._run_tree_search(
        query=retrieval_query,
        knowledge_id=knowledge_id,
    )

if structured:
    unstructured_chunks, structured_chunks = await asyncio.gather(
        unstructured.retrieve(
            query=retrieval_query,
            knowledge_id=knowledge_id,
            tree_chunks=tree_chunks or None,
        ),
        structured.retrieve(query=retrieval_query, knowledge_id=knowledge_id),
    )
    chunks = self._merge_retrieval_results(unstructured_chunks, structured_chunks)
else:
    chunks = await unstructured.retrieve(
        query=retrieval_query,
        knowledge_id=knowledge_id,
        tree_chunks=tree_chunks or None,
    )

if min_score is not None:
    chunks = [c for c in chunks if c.score >= min_score]
return chunks
```

Key change: tree chunks are threaded into the *first and only* `unstructured.retrieve()` call via its existing `tree_chunks` parameter, so fusion happens inside `RetrievalService` where `source_type_weights` and `method_weights` are correctly applied.

**Step 5: Re-run + `poe test` full suite**

```bash
uv run pytest src/rfnry_rag/retrieval/tests/test_tree_search_fusion.py src/rfnry_rag/retrieval/tests/test_server_query.py -v
uv run poe test
```

Both pass.

**Step 6: Commit**

```bash
git add src/rfnry_rag/retrieval/server.py \
        src/rfnry_rag/retrieval/tests/test_tree_search_fusion.py
git commit -m "fix: fuse tree search results via RetrievalService instead of re-invoking retrieve"
```

---

### Task 4: Add `return_exceptions=True` to multi-query and sparse+dense gather sites

**Finding from HIGH correctness set.** Two `asyncio.gather` sites have no exception safety:
- `modules/retrieval/search/service.py:62-63` — one bad query crashes the whole retrieval
- `modules/retrieval/methods/vector.py:131-134` — a sparse embedding failure crashes the vector path

**Files:**
- Modify: `src/rfnry_rag/retrieval/modules/retrieval/search/service.py:62-69`
- Modify: `src/rfnry_rag/retrieval/modules/retrieval/methods/vector.py:131-144`
- Test: `src/rfnry_rag/retrieval/tests/test_gather_exception_safety.py` (new)

**Step 1: Write failing test (multi-query)**

```python
# src/rfnry_rag/retrieval/tests/test_gather_exception_safety.py
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from rfnry_rag.retrieval.common.models import RetrievedChunk
from rfnry_rag.retrieval.modules.retrieval.search.service import RetrievalService


def _chunk(cid: str) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=cid, content=cid, score=1.0, source_id="s", page_number=1)


@pytest.mark.asyncio
async def test_one_failing_query_does_not_crash_retrieval():
    good = SimpleNamespace(
        name="good",
        weight=1.0,
        top_k=None,
        search=AsyncMock(return_value=[_chunk("ok")]),
    )
    rewriter = SimpleNamespace(rewrite=AsyncMock(return_value=["alt1", "alt2"]))

    service = RetrievalService(methods=[good], top_k=5, query_rewriter=rewriter)

    # Simulate one query task raising via a patched _search_single_query
    call_counter = {"n": 0}
    async def flaky(q, *a, **kw):
        call_counter["n"] += 1
        if call_counter["n"] == 2:
            raise RuntimeError("boom")
        return ([[_chunk(f"c{call_counter['n']}")]], [1.0])
    service._search_single_query = flaky  # type: ignore[assignment]

    results = await service.retrieve(query="q")
    # Must not raise; must return the non-failing branches
    assert results
```

**Step 2: Run — expect FAIL** (RuntimeError propagates today).

**Step 3: Fix `search/service.py`**

```python
# Replace lines 62-69 in modules/retrieval/search/service.py
search_tasks = [self._search_single_query(q, fetch_k, filters, knowledge_id) for q in queries]
query_results = await asyncio.gather(*search_tasks, return_exceptions=True)

all_result_lists: list[list[RetrievedChunk]] = []
all_weights: list[float] = []
for idx, outcome in enumerate(query_results):
    if isinstance(outcome, BaseException):
        logger.warning("query variant %d failed: %s — skipping", idx, outcome)
        continue
    result_lists, weights = outcome
    all_result_lists.extend(result_lists)
    all_weights.extend(weights)
```

**Step 4: Fix `methods/vector.py`**

```python
# Replace lines 130-138 in modules/retrieval/methods/vector.py
if self._sparse:
    dense_result, sparse_outcome = await asyncio.gather(
        self._embeddings.embed([query]),
        self._sparse.embed_sparse_query(query),
        return_exceptions=True,
    )
    if isinstance(dense_result, BaseException):
        logger.warning("dense embedding failed: %s", dense_result)
        return []
    if isinstance(sparse_outcome, BaseException):
        logger.warning("sparse embedding failed: %s — falling back to dense only", sparse_outcome)
        sparse_vector = None
    else:
        sparse_vector = sparse_outcome
    query_vector = dense_result[0] if dense_result else None
    if not query_vector:
        logger.warning("embedding returned no vectors for query")
        return []
    if sparse_vector is not None:
        results = await self._store.hybrid_search(
            vector=query_vector, sparse_vector=sparse_vector, top_k=top_k, filters=filters
        )
        logger.info("%d candidates from hybrid search", len(results))
    else:
        results = await self._store.search(vector=query_vector, top_k=top_k, filters=filters)
        logger.info("%d candidates from dense fallback (sparse failed)", len(results))
```

**Step 5: Re-run tests and full suite**

```bash
uv run pytest src/rfnry_rag/retrieval/tests/test_gather_exception_safety.py src/rfnry_rag/retrieval/tests/test_vector_retrieval.py src/rfnry_rag/retrieval/tests/test_hybrid_retrieval.py -v
uv run poe test
```

**Step 6: Commit**

```bash
git add src/rfnry_rag/retrieval/modules/retrieval/search/service.py \
        src/rfnry_rag/retrieval/modules/retrieval/methods/vector.py \
        src/rfnry_rag/retrieval/tests/test_gather_exception_safety.py
git commit -m "fix: isolate per-query and sparse/dense gather failures with return_exceptions"
```

---

### Task 5 (was plan item 3, promoted for sequencing): Promote `bm25_enabled` + `sparse_embeddings` warning to `ConfigurationError`

**Finding.** `server.py:297-298` logs a warning when both are configured, then silently disables BM25 at `server.py:314`. A later config edit that drops `sparse_embeddings` will re-enable BM25 unexpectedly. Validate at config-construction time.

**Files:**
- Modify: `src/rfnry_rag/retrieval/server.py:91-108` (add validation in a new `RagServerConfig.__post_init__`) OR `server.py:237-265` (`_validate_config`)
- Modify: `src/rfnry_rag/retrieval/server.py:291-319` (remove the now-dead warning + `and not bool(...)` guard)
- Test: `src/rfnry_rag/retrieval/tests/test_config_validation.py` (extend existing file)

**Step 1: Write failing test**

Add to `tests/test_config_validation.py`:

```python
def test_bm25_enabled_with_sparse_embeddings_raises():
    from rfnry_rag.retrieval.common.errors import ConfigurationError
    from rfnry_rag.retrieval.server import (
        IngestionConfig, PersistenceConfig, RagEngine, RagServerConfig, RetrievalConfig,
    )
    # Minimal config with both bm25 + sparse configured → must reject
    config = RagServerConfig(
        persistence=PersistenceConfig(vector_store=_DummyVectorStore()),
        ingestion=IngestionConfig(
            embeddings=_DummyEmbeddings(),
            sparse_embeddings=_DummySparseEmbeddings(),
        ),
        retrieval=RetrievalConfig(bm25_enabled=True),
    )
    with pytest.raises(ConfigurationError, match="bm25_enabled .* sparse_embeddings"):
        RagEngine(config)._validate_config()
```

(Re-use existing `_Dummy*` fakes from the same test file.)

**Step 2: Run — FAIL.**

**Step 3: Add to `_validate_config`**

Insert after `server.py:261`:

```python
if cfg.retrieval.bm25_enabled and i.sparse_embeddings:
    raise ConfigurationError(
        "bm25_enabled cannot be used together with sparse_embeddings — "
        "sparse embeddings supersede BM25. Disable one."
    )
```

Remove the now-dead warning at `server.py:297-298` and simplify `server.py:314` to `bm25_enabled=retrieval.bm25_enabled,`.

**Step 4: Re-run + `poe test`.**

**Step 5: Commit**

```bash
git commit -m "fix: reject bm25_enabled + sparse_embeddings at config-validation time"
```

---

## Phase 2 — Modularity (high-ROI)

### Task 6: Export `RetrievalService` and `IngestionService` at package root

**Finding.** Both services accept `list[BaseRetrievalMethod]` / `list[BaseIngestionMethod]` and work fine without `RagEngine`, but they're not in `__all__` so users must do deep imports.

**Files:**
- Modify: `src/rfnry_rag/retrieval/__init__.py:41-149`
- Test: `src/rfnry_rag/retrieval/tests/test_public_api.py` (new)

**Step 1: Write failing test**

```python
# src/rfnry_rag/retrieval/tests/test_public_api.py
def test_retrieval_service_exported_at_package_root():
    import rfnry_rag.retrieval as pkg
    assert "RetrievalService" in pkg.__all__
    assert pkg.RetrievalService is not None


def test_ingestion_service_exported_at_package_root():
    import rfnry_rag.retrieval as pkg
    assert "IngestionService" in pkg.__all__
    assert pkg.IngestionService is not None


def test_semantic_chunker_exported_at_package_root():
    import rfnry_rag.retrieval as pkg
    assert "SemanticChunker" in pkg.__all__


def test_base_protocols_exported():
    import rfnry_rag.retrieval as pkg
    for name in ("BaseRetrievalMethod", "BaseIngestionMethod"):
        assert name in pkg.__all__
```

**Step 2: Run — FAIL** (none of these are exported).

**Step 3: Add to `__init__.py`**

Insert imports + `__all__` entries:

```python
from rfnry_rag.retrieval.modules.ingestion.base import BaseIngestionMethod as BaseIngestionMethod
from rfnry_rag.retrieval.modules.ingestion.chunk.chunker import SemanticChunker as SemanticChunker
from rfnry_rag.retrieval.modules.ingestion.chunk.service import IngestionService as IngestionService
from rfnry_rag.retrieval.modules.retrieval.base import BaseRetrievalMethod as BaseRetrievalMethod
from rfnry_rag.retrieval.modules.retrieval.search.service import RetrievalService as RetrievalService
```

Add to `__all__`:

```python
"RetrievalService",
"IngestionService",
"SemanticChunker",
"BaseRetrievalMethod",
"BaseIngestionMethod",
```

**Step 4: Run — PASS. Commit.**

```bash
git commit -m "feat: export RetrievalService, IngestionService, SemanticChunker, and base protocols"
```

---

### Task 7: Add `RagEngine.vector_only()` / `.document_only()` / `.hybrid()` preset factories

**Finding.** There are no named pipeline presets — every user hand-assembles `RagServerConfig`. Factories that return a pre-wired config remove the discovery cost for the common cases.

**Files:**
- Modify: `src/rfnry_rag/retrieval/server.py` (add classmethods to `RagEngine`)
- Test: `src/rfnry_rag/retrieval/tests/test_engine_presets.py` (new)

**Step 1: Write failing test**

```python
# src/rfnry_rag/retrieval/tests/test_engine_presets.py
from rfnry_rag.retrieval.server import RagEngine, RagServerConfig


def test_vector_only_preset_yields_valid_config(dummy_vector_store, dummy_embeddings):
    config = RagEngine.vector_only(
        vector_store=dummy_vector_store,
        embeddings=dummy_embeddings,
    )
    assert isinstance(config, RagServerConfig)
    assert config.persistence.vector_store is dummy_vector_store
    assert config.ingestion.embeddings is dummy_embeddings
    assert config.persistence.document_store is None
    assert config.persistence.graph_store is None


def test_document_only_preset(dummy_document_store):
    config = RagEngine.document_only(document_store=dummy_document_store)
    assert config.persistence.document_store is dummy_document_store
    assert config.persistence.vector_store is None


def test_hybrid_preset_enables_reranking(
    dummy_vector_store, dummy_embeddings, dummy_document_store, dummy_reranker
):
    config = RagEngine.hybrid(
        vector_store=dummy_vector_store,
        embeddings=dummy_embeddings,
        document_store=dummy_document_store,
        reranker=dummy_reranker,
    )
    assert config.persistence.vector_store is dummy_vector_store
    assert config.persistence.document_store is dummy_document_store
    assert config.retrieval.reranker is dummy_reranker
```

Reuse existing fixtures from `conftest.py` or inline `SimpleNamespace` dummies.

**Step 2: Run — FAIL** (classmethods don't exist).

**Step 3: Add presets to `RagEngine`** (insert after `__init__`):

```python
@classmethod
def vector_only(
    cls,
    *,
    vector_store: BaseVectorStore,
    embeddings: BaseEmbeddings,
    top_k: int = 5,
    reranker: BaseReranking | None = None,
    query_rewriter: BaseQueryRewriting | None = None,
) -> RagServerConfig:
    """Preset: dense vector search only. Add reranker/rewriter for quality."""
    return RagServerConfig(
        persistence=PersistenceConfig(vector_store=vector_store),
        ingestion=IngestionConfig(embeddings=embeddings),
        retrieval=RetrievalConfig(top_k=top_k, reranker=reranker, query_rewriter=query_rewriter),
    )


@classmethod
def document_only(
    cls,
    *,
    document_store: BaseDocumentStore,
    top_k: int = 5,
    reranker: BaseReranking | None = None,
) -> RagServerConfig:
    """Preset: full-text / substring search only. No embeddings."""
    return RagServerConfig(
        persistence=PersistenceConfig(document_store=document_store),
        ingestion=IngestionConfig(),
        retrieval=RetrievalConfig(top_k=top_k, reranker=reranker),
    )


@classmethod
def hybrid(
    cls,
    *,
    vector_store: BaseVectorStore,
    embeddings: BaseEmbeddings,
    document_store: BaseDocumentStore | None = None,
    graph_store: BaseGraphStore | None = None,
    sparse_embeddings: BaseSparseEmbeddings | None = None,
    reranker: BaseReranking | None = None,
    query_rewriter: BaseQueryRewriting | None = None,
    top_k: int = 5,
) -> RagServerConfig:
    """Preset: multi-path retrieval (vector + optional document / graph / sparse) with rerank."""
    return RagServerConfig(
        persistence=PersistenceConfig(
            vector_store=vector_store,
            document_store=document_store,
            graph_store=graph_store,
        ),
        ingestion=IngestionConfig(embeddings=embeddings, sparse_embeddings=sparse_embeddings),
        retrieval=RetrievalConfig(top_k=top_k, reranker=reranker, query_rewriter=query_rewriter),
    )
```

**Step 4: Run + `poe check` + `poe test`.**

**Step 5: Commit**

```bash
git commit -m "feat: add vector_only/document_only/hybrid preset factories on RagEngine"
```

---

### Task 8: Wrap `TreeSearchService` and `StructuredRetrievalService` as `BaseRetrievalMethod` adapters

**Finding.** Tree search and Enrich (structured) aren't plug-compatible with the retrieval method list — they're wired only inside `RagEngine`. Wrapping each in a thin adapter that conforms to `BaseRetrievalMethod` lets users compose them into `RetrievalService(methods=[...])` directly.

**Files:**
- Create: `src/rfnry_rag/retrieval/modules/retrieval/methods/tree.py`
- Create: `src/rfnry_rag/retrieval/modules/retrieval/methods/enrich.py`
- Modify: `src/rfnry_rag/retrieval/__init__.py` (export new classes)
- Test: `src/rfnry_rag/retrieval/tests/test_tree_retrieval_adapter.py`
- Test: `src/rfnry_rag/retrieval/tests/test_enrich_retrieval_adapter.py`

**Step 1: Sketch the protocol shape**

Read `modules/retrieval/base.py` for `BaseRetrievalMethod` signature — it requires `name: str`, `weight: float`, `top_k: int | None`, and `async def search(query, top_k, filters, knowledge_id) -> list[RetrievedChunk]`.

**Step 2: Write failing tests**

```python
# src/rfnry_rag/retrieval/tests/test_tree_retrieval_adapter.py
import pytest
from unittest.mock import AsyncMock


@pytest.mark.asyncio
async def test_tree_retrieval_adapter_conforms_to_base():
    from rfnry_rag.retrieval.modules.retrieval.base import BaseRetrievalMethod
    from rfnry_rag.retrieval.modules.retrieval.methods.tree import TreeRetrieval

    inner = AsyncMock()
    inner.search = AsyncMock(return_value=[])
    adapter = TreeRetrieval(service=inner, weight=0.8, top_k=3)
    assert isinstance(adapter, BaseRetrievalMethod)
    assert adapter.name == "tree"
    assert adapter.weight == 0.8
    assert adapter.top_k == 3
    result = await adapter.search("q", top_k=5, filters=None, knowledge_id=None)
    assert result == []
```

Similar test for `Enrich` adapter in `test_enrich_retrieval_adapter.py`.

**Step 3: Run — FAIL (classes don't exist).**

**Step 4: Implement adapters**

`modules/retrieval/methods/tree.py`:

```python
"""`BaseRetrievalMethod` adapter for tree-based retrieval."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rfnry_rag.retrieval.common.models import RetrievedChunk

if TYPE_CHECKING:
    from rfnry_rag.retrieval.modules.retrieval.tree.service import TreeSearchService


class TreeRetrieval:
    """Adapts `TreeSearchService` to the `BaseRetrievalMethod` protocol."""

    name = "tree"

    def __init__(
        self,
        service: TreeSearchService,
        *,
        weight: float = 1.0,
        top_k: int | None = None,
    ) -> None:
        self._service = service
        self.weight = weight
        self.top_k = top_k

    async def search(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
        knowledge_id: str | None = None,
    ) -> list[RetrievedChunk]:
        effective_top_k = self.top_k if self.top_k is not None else top_k
        try:
            return await self._service.search(
                query=query, top_k=effective_top_k, knowledge_id=knowledge_id
            )
        except Exception as exc:  # per-method error isolation
            from rfnry_rag.retrieval.common.logging import get_logger
            get_logger("retrieval.methods.tree").warning("tree search failed: %s", exc)
            return []
```

`modules/retrieval/methods/enrich.py` follows the same shape, wrapping `StructuredRetrievalService.retrieve(...)`.

**Step 5: Export from `__init__.py`**

Add `TreeRetrieval` and `StructuredRetrieval` (renamed for clarity from `Enrich` — the service itself stays named `StructuredRetrievalService`) to imports + `__all__`.

**Step 6: Run tests, `poe check`, `poe test`. Commit.**

```bash
git commit -m "feat: wrap TreeSearchService and StructuredRetrievalService as BaseRetrievalMethod adapters"
```

> **Note:** `RagEngine.initialize()` still wires these through its own bespoke path. Leave that alone — the adapters are for external users composing pipelines directly.

---

## Phase 3 — Production hardening

### Task 9: Expose pool/timeout knobs on `SQLAlchemyMetadataStore` and `PostgresDocumentStore`

**Finding H1.** Both engines use SQLAlchemy defaults — `pool_size=5, max_overflow=10`, no `pool_pre_ping`, no `pool_recycle`. Idle connections through a firewall will drop.

**Files:**
- Modify: `src/rfnry_rag/retrieval/stores/metadata/sqlalchemy.py` (`__init__` signature)
- Modify: `src/rfnry_rag/retrieval/stores/document/postgres.py` (`__init__` signature)
- Test: existing `test_postgres_document_store.py` + add coverage in `test_metadata_store_migration.py`

**Step 1: Write failing test**

```python
def test_sqlalchemy_store_accepts_pool_knobs(tmp_path):
    from rfnry_rag.retrieval.stores.metadata.sqlalchemy import SQLAlchemyMetadataStore
    store = SQLAlchemyMetadataStore(
        f"sqlite:///{tmp_path / 't.db'}",
        pool_size=10,
        max_overflow=20,
        pool_recycle=1800,
        pool_pre_ping=True,
    )
    # SQLite uses StaticPool so pool_size is irrelevant, but the call must not error
    assert store._engine is not None
```

And the postgres equivalent (using a fake URL — engine construction is lazy).

**Step 2: Extend `__init__` on both classes**

```python
def __init__(
    self,
    url: str,
    *,
    pool_size: int | None = None,
    max_overflow: int | None = None,
    pool_recycle: int = 1800,
    pool_pre_ping: bool = True,
    echo: bool = False,
) -> None:
    parsed = make_url(url)
    # ... driver normalization as before ...
    kwargs: dict[str, Any] = {"echo": echo, "pool_pre_ping": pool_pre_ping, "pool_recycle": pool_recycle}
    if pool_size is not None:
        kwargs["pool_size"] = pool_size
    if max_overflow is not None:
        kwargs["max_overflow"] = max_overflow
    self._engine = create_async_engine(parsed, **kwargs)
    ...
```

Note: SQLite + `aiosqlite` does not support `pool_size` — guard by checking `parsed.drivername` and only apply pool params when `postgresql+asyncpg`.

**Step 3: Document the new params in the main README (link from retrieval/README.md).** Not in code comments — in the user-facing config table.

**Step 4: Commit**

```bash
git commit -m "feat: expose pool_size/max_overflow/pool_recycle/pool_pre_ping on SQL stores"
```

---

### Task 10: Cap BM25 corpus memory and make `split_pdf_to_images` a generator

**Finding H2 + H3.** BM25 loads the full Qdrant collection into RAM unbounded; PDF analyze loads every page as base64 PNG before processing the first one.

This is two related commits — keep them separate so they bisect cleanly.

**10a: BM25 cap**

- Modify: `src/rfnry_rag/retrieval/modules/retrieval/methods/vector.py` (around `_build_bm25_index`, lines ~282-305)
- Modify: `src/rfnry_rag/retrieval/server.py` `RetrievalConfig` to add `bm25_max_chunks: int = 50_000`
- Test: extend `test_vector_retrieval.py` with a bounded-cap test

Steps: add failing test asserting `_build_bm25_index` stops scrolling at `bm25_max_chunks` and logs a warning. Implement. Commit as `feat: add bm25_max_chunks cap to prevent memory blow-up on large collections`.

**10b: PDF image generator**

- Modify: `src/rfnry_rag/retrieval/modules/ingestion/analyze/pdf_splitter.py` — convert `split_pdf_to_images` to an async generator yielding `(page_number, base64_png)` tuples
- Modify: `src/rfnry_rag/retrieval/modules/ingestion/analyze/service.py:~295-297` — consume the generator lazily
- Test: `test_pdf_splitter_generator.py` (new) — assert only one page is materialized at a time by wrapping with a counter

Commit as `refactor: stream PDF pages one at a time during analyze to cap memory`.

---

### Task 11: Bound `BatchIngestionService.ingest_stream` task creation

**Finding H4.** Tasks are scheduled for every batch before gathering — for a million-record stream this creates 10K task objects simultaneously.

**Files:**
- Modify: `src/rfnry_rag/retrieval/modules/ingestion/chunk/batch.py:135-148`
- Test: `src/rfnry_rag/retrieval/tests/test_batch_ingestion_backpressure.py` (new)

**Step 1: Failing test** — feed 10 batches, asserting that no more than `max_inflight = concurrency * 2` tasks are scheduled at once. Use an `asyncio.Semaphore` counter wrapped around `_process_batch`.

**Step 2: Implement** — replace the "create all tasks then gather" pattern with a bounded producer:

```python
async def ingest_stream(self, records_iter, max_inflight: int | None = None):
    max_inflight = max_inflight or self._config.concurrency * 2
    in_progress: set[asyncio.Task] = set()
    ...
    async for batch in self._batched(records_iter):
        if len(in_progress) >= max_inflight:
            done, in_progress = await asyncio.wait(in_progress, return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                self._aggregate_batch_stats(t.result())
        task = asyncio.create_task(self._process_batch(batch))
        in_progress.add(task)
    # drain
    if in_progress:
        await asyncio.gather(*in_progress)
```

**Step 3: Run + commit.**

```bash
git commit -m "fix: bound inflight batch tasks in BatchIngestionService.ingest_stream"
```

---

### Task 12: Add timeouts + HTTP limits to Qdrant and Neo4j clients; stop mutating `os.environ`

Three small sub-commits.

**12a: Qdrant timeouts**

- Modify: `src/rfnry_rag/retrieval/stores/vector/qdrant.py:62` — accept `timeout: float = 10.0` and pass to `AsyncQdrantClient(url=..., timeout=timeout)`
- Modify: `_ensure_and_resolve` (line ~121) — raise `StoreError` on connection errors instead of returning `None` silently
- Test: add to existing store test

Commit: `feat: add configurable timeout to Qdrant client and surface connection errors`

**12b: Neo4j timeouts + batched writes**

- Modify: `src/rfnry_rag/retrieval/stores/graph/neo4j.py:172-175` — accept `max_connection_pool_size: int = 100`, `connection_acquisition_timeout: float = 60.0`
- Modify: ingestion loop (lines ~196-208) — use `session.execute_write(tx_fn)` with a batch of entities per transaction (batch size 100)
- Test: extend `test_neo4j_graph_store.py`

Commit: `feat: add Neo4j pool/timeout knobs and batch entity writes`

**12c: Stop mutating `os.environ["BOUNDARY_API_KEY"]`**

- Modify: `src/rfnry_rag/retrieval/common/language_model.py:101` (and mirror in `reasoning/common/language_model.py` if shared) — pass the boundary key to BAML's `ClientOptions` or registry directly instead of writing to `os.environ`
- Test: a new test that creates two `LanguageModelClient` instances with different `boundary_api_key` values and asserts neither process-env is clobbered

Commit: `fix: pass boundary_api_key via BAML client options instead of mutating os.environ`

---

### Task 13: Validate TOML config keys + chmod `config.toml` + gate query logging

Three small independent changes.

**13a: Reject unknown TOML keys**

- Modify: `src/rfnry_rag/retrieval/cli/config.py` (around the TOML load at ~239-315)
- Add: after `tomllib.load(f)`, compare top-level keys against `{"persistence", "ingestion", "retrieval", "generation", "tree_indexing", "tree_search"}`. Unknowns → raise `ConfigError` with the full list (helps with typos).
- Also compare each section's keys against the corresponding dataclass field names.
- Test: `src/rfnry_rag/retrieval/tests/test_cli_config_validation.py` (new)

Commit: `fix: reject unknown keys in config.toml to catch typos early`

**13b: `chmod 0o600` for `config.toml`**

- Modify: `src/rfnry_rag/retrieval/cli/commands/init.py:44` (where `.env` is chmodded) — mirror the call for `config.toml`
- Test: extend existing init test to assert `config.toml.stat().st_mode & 0o777 == 0o600`

Commit: `fix: restrict config.toml permissions to 0o600 at init time`

**13c: Gate query text logging behind explicit env var**

- Modify: `src/rfnry_rag/retrieval/modules/retrieval/search/service.py:46` — wrap the `logger.info('query: "%s"', ...)` in `if os.environ.get("RFNRY_RAG_LOG_QUERIES") == "true":`
- Always log the knowledge_id and query length; log the text only under the flag
- Add doc line to the main README under "Environment Variables"
- Test: `test_query_logging_gated.py` — stub the logger, assert no query text is logged when env is unset

Commit: `fix: gate user query text logging behind RFNRY_RAG_LOG_QUERIES env var`

---

## Phase 4 — Nice-to-haves

### Task 14: Tree-index cleanup + stale-source API + graph-store LLM coupling

**14a: Clean up tree indexes on source removal**

- Modify: `src/rfnry_rag/retrieval/server.py:_on_source_removed` (around line 752-757) — after BM25 invalidation, call `metadata_store.delete_tree_index(source_id)` if the method exists
- Add: `delete_tree_index` method on `BaseMetadataStore` + `SQLAlchemyMetadataStore` implementation
- Test: `test_source_removal_cleans_tree_index.py` (new) — create a source with a tree index, remove the source, assert tree index is gone

Commit: `fix: delete tree index when source is removed to prevent orphans`

**14b: Expose stale-source API**

- Modify: `src/rfnry_rag/retrieval/modules/knowledge/manager.py` — add:
  - `async def list_stale(self) -> list[Source]` — returns sources whose `embedding_model_name` differs from current
  - `async def purge_stale(self) -> int` — removes and returns count
- Modify: `__init__.py` — document in the public API reference table in `retrieval/README.md`
- Test: `test_knowledge_stale.py` (new)

Commit: `feat: add KnowledgeManager.list_stale and purge_stale for embedding-migration cleanup`

**14c: Decouple graph-store from `ingestion.lm_client`**

- Modify: `src/rfnry_rag/retrieval/server.py:259-261` — replace the unconditional "graph_store requires ingestion.lm_client" check with: only require `lm_client` if graph ingestion will actually run. A retrieval-only user with a pre-populated graph should not be forced to configure an LLM.
- Practically: leave the check but gate it on whether `GraphIngestion` will be instantiated, or move it into the path where `GraphIngestion` is actually appended to `ingestion_methods` (~line 285 in `initialize`).
- Test: extend `test_config_validation.py` — a config with `graph_store` but no `ingestion.lm_client` should pass validation *if* the user only intends to retrieve.

Trade-off note: this relaxes a safety check. The safest version leaves validation in `_validate_config` but only errors when an ingestion operation is attempted, not at engine-init time. Prefer deferring to a documented "graph ingestion requires lm_client" runtime error.

Commit: `fix: allow retrieval-only graph configuration without ingestion.lm_client`

---

## Final verification

After all 14 task groups commit:

```bash
uv run poe check && uv run poe typecheck && uv run poe test:cov
```

All green. Run one manual smoke with the CLI against a small fixture corpus:

```bash
rfnry-rag retrieval init
rfnry-rag retrieval ingest examples/retrieval/cli/sample.pdf -k smoke
rfnry-rag retrieval query "what is the subject of this document?"
```

Assert: no credential strings in logs, no unhandled exception on a flaky sparse provider, preset factories construct valid configs, BM25 cap log appears on a large collection.

---

## Execution Handoff

**Two execution options:**

**1. Subagent-Driven (this session)** — I dispatch a fresh subagent per task with code review between them. Use for tight feedback and mid-course corrections.

**2. Parallel Session (separate)** — Open a new session in a worktree, point it at this plan, and use `superpowers:executing-plans` for batch execution with checkpoints. Use if you want to keep this session free for other work.

Which approach?
