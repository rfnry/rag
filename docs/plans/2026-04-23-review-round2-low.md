# Review Round 2 — Low-Priority Cleanup

> **For Claude:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task.

**Goal:** Resolve the LOW findings from the 2026-04-23 second-pass review of the `rfnry-rag` SDK. None of these individually block production rollout, but together they eliminate papercuts, pattern traps, and minor consistency drift.

**Architecture:** Grouped by theme (naming, config safety, performance, security) rather than severity. Each group shares context; tasks within a group are independent commits. TDD-first where behavior changes; doc-only tasks get a verification command instead of a test.

**Tech stack:** Python 3.12, pytest (`asyncio_mode=auto`), Ruff, MyPy. Run from `packages/python/`.

**Preconditions:**
- The critical/high/medium plan (`2026-04-23-review-round2-critical-high-medium.md`) is fully merged — several items here depend on fixes from that plan (e.g. Task 11's parallel dispatch changes `_dispatch_methods`, which some LOW tasks touch).
- `uv run poe test` green on main.

**Verification after every task:**
```bash
uv run poe check
uv run poe typecheck
uv run poe test
```

---

## Group A — Naming & API hygiene

### Task L1 — Rename internal `contextual_chunking` param to match public config

**Finding.** Public `IngestionConfig.chunk_context_headers` was renamed with a deprecation shim for the old `contextual_chunking` field. But `IngestionService.__init__(..., contextual_chunking: bool = True, ...)` at `src/rfnry_rag/retrieval/modules/ingestion/chunk/service.py:56,65,189,250` still uses the old name. Call sites at `server.py:563, 1041` and `cli/config.py:283` still pass `contextual_chunking=...`. Public name diverges from internal.

**Files:**
- Modify: `src/rfnry_rag/retrieval/modules/ingestion/chunk/service.py` — rename constructor param and instance attr.
- Modify: `src/rfnry_rag/retrieval/server.py:563, 1041` — kwarg rename.
- Modify: `src/rfnry_rag/retrieval/cli/config.py:283` — kwarg rename (keep reading `contextual_chunking` from TOML with the deprecation shim, but pass the new name downstream).
- Modify: `src/rfnry_rag/retrieval/tests/test_ingestion_service_methods.py:234, 263, 294` and `test_ingestion_required_methods.py:42, 69, 101` — rename kwargs in test helpers.

**Step 1 — Test:** Existing tests are the regression guard — the old name is gone, the new name must work.

**Step 2 — Implementation:**
- Rename constructor parameter `contextual_chunking` → `chunk_context_headers` in `IngestionService.__init__`.
- Rename instance attribute `self._contextual_chunking` → `self._chunk_context_headers`. Update both read sites (`:189, :250`).
- Update all internal callers.
- In `cli/config.py`, keep the TOML key `contextual_chunking` readable (the deprecation shim in `IngestionConfig.__post_init__` handles that) but pass `chunk_context_headers=` to any direct `IngestionService` construction.

**Step 3 — Verify:**
```bash
grep -rn "contextual_chunking" src/ | grep -v baml_client | grep -v test_config_validation.py | grep -v "DEPRECATED\|Deprecated\|deprecated"
```
Expected: only references inside the deprecation shim and its tests.

**Step 4 — Commit:**
```bash
git commit -m "refactor: rename IngestionService.contextual_chunking to chunk_context_headers"
```

---

### Task L2 — Thread `MethodNamespace[T]` type param through call sites

**Finding.** `src/rfnry_rag/retrieval/modules/namespace.py:26` — `MethodNamespace[T].__iter__` returns `Any`. Properties `RagEngine.retrieval` / `RagEngine.ingestion` at `server.py:341, 365, 372` declare bare `MethodNamespace`. The generic is effectively decorative.

**Files:**
- Modify: `src/rfnry_rag/retrieval/modules/namespace.py`
- Modify: `src/rfnry_rag/retrieval/server.py` properties `retrieval` and `ingestion` type annotations.

**Step 1 — Test:** Add an mypy expectation test (or just rely on `poe typecheck`). Example in a new `test_namespace_typing.py`:

```python
# typing-only test — relies on mypy catching regressions
# No assert needed; mypy --strict would check
```

Better: write a concrete test that exercises iteration and confirms the type is preserved:

```python
from rfnry_rag.retrieval.modules.namespace import MethodNamespace
from rfnry_rag.retrieval.modules.retrieval.base import BaseRetrievalMethod
from unittest.mock import MagicMock

def test_method_namespace_iter_preserves_type_param() -> None:
    method = MagicMock(spec=BaseRetrievalMethod)
    method.name = "vector"
    ns: MethodNamespace[BaseRetrievalMethod] = MethodNamespace([method])
    items = list(ns)
    assert len(items) == 1
    # Type-level check is via mypy; runtime assertion simply exercises the iterator.
```

**Step 2 — Implementation:**

In `namespace.py`:
```python
from collections.abc import Iterator
class MethodNamespace(Generic[T]):
    ...
    def __iter__(self) -> Iterator[T]:
        return iter(self._methods)
```

In `server.py`:
```python
@property
def retrieval(self) -> MethodNamespace[BaseRetrievalMethod]:
    ...

@property
def ingestion(self) -> MethodNamespace[BaseIngestionMethod]:
    ...
```

Make sure imports of `BaseRetrievalMethod` and `BaseIngestionMethod` are available at the property-annotation level (they may already be `TYPE_CHECKING`-guarded; move to runtime import or use string annotations if it causes circular issues).

**Step 3 — Verify:** `uv run poe typecheck` stays clean.

**Step 4 — Commit:**
```bash
git commit -m "refactor: thread MethodNamespace type parameter through engine properties"
```

---

### Task L3 — Deduplicate `_synthesize_l5x` / `_synthesize_xml` loop bodies

**Finding.** `src/rfnry_rag/retrieval/modules/ingestion/analyze/service.py:409-459` — both methods share identical shared-entity cross-reference logic.

**Files:**
- Modify: `src/rfnry_rag/retrieval/modules/ingestion/analyze/service.py:409-459`
- Test: `src/rfnry_rag/retrieval/tests/test_analyzed_ingestion.py` (verify both paths still work after extraction)

**Step 1 — Run existing tests green (baseline).**

**Step 2 — Implementation:** Extract the shared loop body into a private `_synthesize_shared_entities(...)` helper. Both `_synthesize_l5x` and `_synthesize_xml` call the helper. The helper's signature should be whatever set of parameters the shared loop uses — read both methods and pick the intersection.

**Step 3 — Verify tests still pass.**

**Step 4 — Commit:**
```bash
git commit -m "refactor: extract shared synthesise-entities loop body in analyzed ingestion"
```

---

## Group B — Config safety (visibility & bounds)

### Task L4 — Expose `fetch_k` hybrid-search prefetch multiplier

**Finding.** `src/rfnry_rag/retrieval/stores/vector/qdrant.py:228` — `fetch_k = top_k * 4`. At `top_k=200`, prefetches 1600 candidates per leg (3200 pre-fusion). Hardcoded magic number.

**Files:**
- Modify: `src/rfnry_rag/retrieval/stores/vector/qdrant.py`
- Test: `src/rfnry_rag/retrieval/tests/test_store_pool_knobs.py` (extend)

**Step 1 — Failing test:**

```python
async def test_qdrant_hybrid_prefetch_multiplier_is_configurable() -> None:
    store = QdrantVectorStore(url="http://fake", hybrid_prefetch_multiplier=8)
    # Patch _ensure_and_resolve and _client to capture prefetch limit.
    ...
    # Assert that hybrid_search used fetch_k = top_k * 8.
```

**Step 2 — Red.**

**Step 3 — Implementation:**

```python
def __init__(self, ..., hybrid_prefetch_multiplier: int = 4, ...):
    if hybrid_prefetch_multiplier < 1:
        raise ConfigurationError("hybrid_prefetch_multiplier must be >= 1")
    self._hybrid_prefetch_multiplier = hybrid_prefetch_multiplier

# In hybrid_search:
fetch_k = top_k * self._hybrid_prefetch_multiplier
```

**Step 4 — Green.**

**Step 5 — Commit:**
```bash
git commit -m "config: expose hybrid_prefetch_multiplier on QdrantVectorStore"
```

---

### Task L5 — Expose `ts_headline` knobs on `PostgresDocumentStore`

**Finding.** `src/rfnry_rag/retrieval/stores/document/postgres.py:163` — `'MaxWords=200,MinWords=80,MaxFragments=3'` hardcoded as an SQL literal.

**Files:**
- Modify: `src/rfnry_rag/retrieval/stores/document/postgres.py`

**Step 1 — Failing test:**

```python
async def test_postgres_headline_options_configurable() -> None:
    store = PostgresDocumentStore(url="...", headline_max_words=50, headline_min_words=20, headline_max_fragments=1)
    # Execute a query and capture the generated SQL (via an event listener
    # or by inspecting the compiled query). Assert "MaxWords=50" etc.
    ...
```

**Step 2 — Red.**

**Step 3 — Implementation:**

Add three constructor params with the current defaults. Build the option string at query time:

```python
headline_opts = f"MaxWords={self._headline_max_words},MinWords={self._headline_min_words},MaxFragments={self._headline_max_fragments}"
```

(If Task 15 from the higher-priority plan converted this to SQLAlchemy Core, pass the string as a `literal(...)` argument — otherwise keep the existing text/f-string shape but parameterised via constructor.)

**Step 4 — Green.**

**Step 5 — Commit:**
```bash
git commit -m "config: expose ts_headline MaxWords/MinWords/MaxFragments on PostgresDocumentStore"
```

---

### Task L6 — Expose `_build_retrieval_query` history window size

**Finding.** `src/rfnry_rag/retrieval/server.py:934` — hardcoded `history[-3:]`. Long human turns + 3 history entries can push the enriched query toward the 32k-char limit.

**Files:**
- Modify: `src/rfnry_rag/retrieval/server.py:924-940` (static method — consider passing via config)
- Modify: `RetrievalConfig` — add `history_window: int = 3`

**Step 1 — Failing test:**

```python
async def test_retrieval_uses_configured_history_window() -> None:
    cfg = RetrievalConfig(history_window=1)
    assert cfg.history_window == 1
    # Full engine wiring test: assert that with history_window=1, only the last
    # turn is appended.
```

**Step 2 — Red.**

**Step 3 — Implementation:** Add `history_window: int = 3` to `RetrievalConfig` with a `__post_init__` bound (`1 <= history_window <= 20`). Change `_build_retrieval_query` from `@staticmethod` to an instance method that reads `self._config.retrieval.history_window`.

**Step 4 — Green.**

**Step 5 — Commit:**
```bash
git commit -m "config: expose history_window on RetrievalConfig (was hardcoded history[-3:])"
```

---

### Task L7 — Validate `RFNRY_RAG_LOG_LEVEL` against real level names

**Finding.** `src/rfnry_rag/common/logging.py:31` — an invalid value (e.g. `TRACE`) silently sets level 0 (NOTSET), enabling all output.

**Files:**
- Modify: `src/rfnry_rag/common/logging.py`
- Test: `src/rfnry_rag/common/tests/test_logging.py` or closest

**Step 1 — Failing test:**

```python
def test_invalid_log_level_raises_configuration_error(monkeypatch) -> None:
    monkeypatch.setenv("RFNRY_RAG_LOG_LEVEL", "TRACE")
    monkeypatch.setenv("RFNRY_RAG_LOG_ENABLED", "true")
    # Force re-init of the logger
    with pytest.raises(ConfigurationError, match="unknown log level"):
        get_logger("test").info("x")
```

**Step 2 — Red.**

**Step 3 — Implementation:**

```python
import logging
_VALID_LEVELS = set(logging.getLevelNamesMapping())

def _resolve_level(raw: str) -> int:
    upper = raw.upper()
    if upper not in _VALID_LEVELS:
        raise ConfigurationError(
            f"unknown log level {raw!r}; valid: {sorted(_VALID_LEVELS)}"
        )
    return logging.getLevelNamesMapping()[upper]
```

**Step 4 — Green.**

**Step 5 — Commit:**
```bash
git commit -m "config: validate RFNRY_RAG_LOG_LEVEL against logging.getLevelNamesMapping()"
```

---

### Task L8 — Log effective pool sizes and retry policy at startup

**Finding.** Operators can't confirm from logs whether `pool_size` / `max_overflow` / `LanguageModelClient.max_retries` / `fallback` are the values they configured or library defaults.

**Files:**
- Modify: `src/rfnry_rag/retrieval/stores/metadata/sqlalchemy.py` — log effective pool settings in `initialize()`.
- Modify: `src/rfnry_rag/retrieval/stores/document/postgres.py` — same.
- Modify: `src/rfnry_rag/common/language_model.py::build_registry` — log effective `max_retries`, `strategy`, `timeout_seconds`, and whether a fallback is wired.

**Step 1 — Failing test (one form):**

```python
async def test_metadata_store_logs_effective_pool_size(caplog) -> None:
    caplog.set_level("INFO", logger="rfnry_rag.retrieval.stores.metadata.sqlalchemy")
    store = SQLAlchemyMetadataStore(url="sqlite:///:memory:", pool_size=7, max_overflow=14)
    await store.initialize()
    messages = "\n".join(r.message for r in caplog.records)
    # sqlite uses StaticPool — log should mention that explicitly
    assert "pool" in messages.lower()
```

For postgres/pg URLs, parameterise similarly but keep the test non-network-dependent (log whatever the store *would* configure).

**Step 2 — Red.**

**Step 3 — Implementation:** Add `logger.info("sqlalchemy metadata store: pool_size=%s max_overflow=%s pool_recycle=%ds pool_timeout=%ds", ...)` to `initialize()`. For SQLite-only drivers, log `"sqlite driver: static pool (pool_size/max_overflow not applicable)"`.

In `build_registry`, add:
```python
logger.info(
    "language model client: provider=%s model=%s strategy=%s max_retries=%d timeout=%ds fallback=%s",
    client.provider.provider, client.provider.model, client.strategy, client.max_retries,
    client.timeout_seconds, bool(client.fallback),
)
```

**Step 4 — Green.**

**Step 5 — Commit:**
```bash
git commit -m "observability: log effective pool sizes and LLM client policy at startup"
```

---

### Task L9 — Upper-bound `run_concurrent` concurrency argument

**Finding.** `src/rfnry_rag/common/concurrency.py:5-18` — `run_concurrent(concurrency=N)` accepts any integer. `BatchConfig.concurrency` validates `1-20` but callers bypassing `BatchConfig` have no guard.

**Files:**
- Modify: `src/rfnry_rag/common/concurrency.py`
- Test: `src/rfnry_rag/common/tests/test_concurrency.py` (extend or create)

**Step 1 — Failing test:**

```python
async def test_run_concurrent_rejects_nonpositive_concurrency() -> None:
    with pytest.raises(ValueError):
        await run_concurrent([1], fn=AsyncMock(), concurrency=0)
    with pytest.raises(ValueError):
        await run_concurrent([1], fn=AsyncMock(), concurrency=-1)


async def test_run_concurrent_rejects_excessive_concurrency() -> None:
    with pytest.raises(ValueError, match="concurrency must be <= "):
        await run_concurrent([1], fn=AsyncMock(), concurrency=10_000)
```

**Step 2 — Red.**

**Step 3 — Implementation:**

```python
_MAX_CONCURRENCY = 100    # hard upper bound; any caller hitting this needs a semaphore of its own

async def run_concurrent[T, R](items, fn, concurrency):
    if concurrency < 1:
        raise ValueError(f"concurrency must be >= 1, got {concurrency}")
    if concurrency > _MAX_CONCURRENCY:
        raise ValueError(f"concurrency must be <= {_MAX_CONCURRENCY}, got {concurrency}")
    ...
```

**Step 4 — Green.**

**Step 5 — Commit:**
```bash
git commit -m "safety: bound run_concurrent concurrency argument (1 <= n <= 100)"
```

---

### Task L10 — Reverse-init shutdown order + clear service refs

**Finding.** `src/rfnry_rag/retrieval/server.py:677-701` — shutdown order is `vector → metadata → document → graph`, init order is `metadata → document → graph → vector`. Reverse-init would be safer for future shutdown hooks that cross-read stores. Also `self._initialized = False` but service references (`self._generation_service`, etc.) still point to objects backed by closed clients.

**Files:**
- Modify: `src/rfnry_rag/retrieval/server.py:677-701`

**Step 1 — Failing test:**

```python
async def test_shutdown_tears_down_in_reverse_init_order() -> None:
    calls: list[str] = []
    mock_metadata = _mock_store_tracking(calls, "metadata")
    mock_document = _mock_store_tracking(calls, "document")
    mock_graph = _mock_store_tracking(calls, "graph")
    mock_vector = _mock_store_tracking(calls, "vector")
    rag = _build_rag_with_all_stores(metadata=mock_metadata, document=mock_document,
                                     graph=mock_graph, vector=mock_vector)
    await rag.initialize()
    calls.clear()
    await rag.shutdown()
    # Init order was metadata, document, graph, vector → reverse for shutdown
    assert calls == ["vector", "graph", "document", "metadata"]


async def test_shutdown_clears_service_references() -> None:
    rag = await _build_initialized_rag()
    await rag.shutdown()
    assert rag._generation_service is None
    assert rag._retrieval_service is None
    assert rag._ingestion_service is None
```

**Step 2 — Red.**

**Step 3 — Implementation:** Reorder shutdown to reverse init. After all shutdowns, set service refs to `None`:

```python
async def shutdown(self) -> None:
    persistence = self._config.persistence
    # Reverse-init order: vector was initialized last
    if persistence.vector_store:
        try: await persistence.vector_store.shutdown()
        except Exception: logger.exception("error shutting down vector store")
    if persistence.graph_store:
        try: await persistence.graph_store.shutdown()
        except Exception: logger.exception("error shutting down graph store")
    if persistence.document_store:
        try: await persistence.document_store.shutdown()
        except Exception: logger.exception("error shutting down document store")
    if persistence.metadata_store:
        try: await persistence.metadata_store.shutdown()
        except Exception: logger.exception("error shutting down metadata store")

    self._ingestion_service = None
    self._structured_ingestion = None
    self._retrieval_service = None
    self._structured_retrieval = None
    self._generation_service = None
    self._knowledge_manager = None
    self._step_service = None
    self._tree_indexing_service = None
    self._tree_search_service = None
    self._retrieval_namespace = None
    self._ingestion_namespace = None
    self._retrieval_by_collection.clear()
    self._ingestion_by_collection.clear()

    self._initialized = False
    logger.info("ragengine shut down")
```

**Step 4 — Green.**

**Step 5 — Commit:**
```bash
git commit -m "lifecycle: shut down stores in reverse-init order; clear service refs"
```

---

## Group C — Performance (micro)

### Task L11 — Hold lock around `_bm25_search` initial cache read

**Finding.** `src/rfnry_rag/retrieval/modules/retrieval/methods/vector.py:244` — initial `if key not in self._bm25_cache` check is outside the lock. Concurrent `invalidate_cache` (which *does* hold the lock) can race, causing a torn read of the entry dict.

**Files:**
- Modify: `src/rfnry_rag/retrieval/modules/retrieval/methods/vector.py`

**Step 1 — Failing test:** Hard to produce deterministically in asyncio. Skip; the fix is audit-verified. Rely on existing BM25 tests to confirm no regression.

**Step 2 — Implementation:**

Read the entry under the lock once:

```python
async def _bm25_search(self, query, top_k, knowledge_id) -> list[RetrievedChunk]:
    key = knowledge_id if knowledge_id is not None else _GLOBAL_KEY
    async with self._bm25_lock:
        entry = self._bm25_cache.get(key)
    if entry is None:
        await self._build_bm25_index(knowledge_id)
        async with self._bm25_lock:
            entry = self._bm25_cache.get(key)
    if entry is None or entry.index is None or not entry.chunks:
        return []
    entry.last_used = time.monotonic()
    ...
```

(The `last_used` write is still non-atomic w.r.t. the lock, but it's a single attribute assignment on a cached entry and asyncio is cooperative — safe without extra locking.)

**Step 3 — Run BM25 tests to verify no regression.**

**Step 4 — Commit:**
```bash
git commit -m "safety: hold BM25 lock around initial cache read to prevent torn read vs invalidate"
```

---

### Task L12 — Push duplicate-hash check into the metadata store

**Finding.** `src/rfnry_rag/retrieval/modules/ingestion/chunk/service.py:79-87` — `_check_duplicate` lists all sources for a knowledge_id and scans in Python. O(N) per ingest.

**Files:**
- Modify: `src/rfnry_rag/retrieval/stores/metadata/base.py` — add a new protocol method `find_by_hash`.
- Modify: `src/rfnry_rag/retrieval/stores/metadata/sqlalchemy.py` — implement.
- Modify: `src/rfnry_rag/retrieval/modules/ingestion/chunk/service.py` — use it.

**Step 1 — Failing test:**

```python
async def test_metadata_store_find_by_hash_returns_matching_source() -> None:
    store = _sqlalchemy_store_fixture(":memory:")
    await store.initialize()
    src = await store.create_source(source_id="s1", knowledge_id="kb1", file_hash="abc", ...)
    found = await store.find_by_hash(hash_value="abc", knowledge_id="kb1")
    assert found is not None and found.source_id == "s1"
    assert await store.find_by_hash(hash_value="nope", knowledge_id="kb1") is None
```

**Step 2 — Red.**

**Step 3 — Implementation:**

In `base.py` Protocol:
```python
async def find_by_hash(self, hash_value: str, knowledge_id: str | None) -> Source | None: ...
```

In `sqlalchemy.py`:
```python
async def find_by_hash(self, hash_value: str, knowledge_id: str | None) -> Source | None:
    stmt = select(_SourceRow).where(_SourceRow.file_hash == hash_value)
    if knowledge_id is not None:
        stmt = stmt.where(_SourceRow.knowledge_id == knowledge_id)
    stmt = stmt.limit(1)
    async with self._session_factory() as session:
        row = (await session.execute(stmt)).scalars().first()
        return None if row is None else _row_to_source(row)
```

In `service.py::_check_duplicate`:
```python
async def _check_duplicate(self, hash_value: str, knowledge_id: str | None) -> None:
    if not self._metadata_store:
        return
    existing = await self._metadata_store.find_by_hash(hash_value, knowledge_id)
    if existing is not None:
        raise DuplicateSourceError(
            f"File already ingested as source {existing.source_id} (hash={hash_value[:12]}...)"
        )
```

**Step 4 — Green.**

**Step 5 — Commit:**
```bash
git commit -m "perf: push duplicate-hash check into metadata store (SQL WHERE instead of Python scan)"
```

---

## Group D — Security (small surfaces)

### Task L13 — `generate_step()` must validate query length

**Finding.** `src/rfnry_rag/retrieval/server.py:887-906` — `generate_step(query=...)` is the only public method accepting a `query` string that skips `_validate_query_text`.

**Files:**
- Modify: `src/rfnry_rag/retrieval/server.py:897`

**Step 1 — Failing test:**

```python
async def test_generate_step_rejects_oversize_query() -> None:
    rag = await _build_initialized_rag_with_step()
    with pytest.raises(ValueError, match="query exceeds"):
        await rag.generate_step(query="x" * 40_000, chunks=[])
```

**Step 2 — Red.**

**Step 3 — Implementation:** Add `_validate_query_text(query)` immediately after `self._check_initialized()` in `generate_step`.

**Step 4 — Green.**

**Step 5 — Commit:**
```bash
git commit -m "security: validate query length in RagEngine.generate_step"
```

---

### Task L14 — Escape filesystem frontmatter `title`

**Finding.** `src/rfnry_rag/retrieval/stores/document/filesystem.py:83-86` — frontmatter writes `title: {title}` unescaped. A title containing `\n---\n` corrupts the parser at line 208. Data integrity issue, not path traversal.

**Files:**
- Modify: `src/rfnry_rag/retrieval/stores/document/filesystem.py`

**Step 1 — Failing test:**

```python
async def test_filesystem_store_handles_title_with_frontmatter_delimiter(tmp_path) -> None:
    store = FilesystemDocumentStore(base_path=str(tmp_path))
    await store.initialize()
    bad_title = "normal start\n---\nend: injected"
    await store.store_content(source_id="s1", knowledge_id=None, source_type=None,
                              title=bad_title, content="hello")
    hits = await store.search("hello")
    assert len(hits) == 1
    assert hits[0].title == bad_title   # round-trip preserved
```

**Step 2 — Red.**

**Step 3 — Implementation:** Use a safe serialisation for frontmatter values. Options:
- JSON-encode the value: `json.dumps(title)` gives `"normal start\n---\nend: injected"` which survives line-based parsing.
- Switch to real YAML (adds a dep — skip).

Pick JSON encoding for strings and apply consistently on read. In the reader at line 208, parse with `json.loads` when the value is JSON-shaped; otherwise use raw. Simplest: always JSON-encode on write, always `json.loads` on read.

**Step 4 — Green.**

**Step 5 — Commit:**
```bash
git commit -m "fix: JSON-encode filesystem frontmatter values to survive embedded delimiters"
```

---

### Task L15 — Bound reasoning CLI directory-read size

**Finding.** `src/rfnry_rag/reasoning/cli/__init__.py:46` — `for f in path.iterdir(): parts.append(f"[{f.name}]\n{f.read_text()}")`. No per-file or aggregate size gate before LLM send.

**Files:**
- Modify: `src/rfnry_rag/reasoning/cli/__init__.py`

**Step 1 — Failing test:**

```python
def test_reasoning_cli_directory_read_caps_aggregate_size(tmp_path) -> None:
    big = tmp_path / "big.txt"
    big.write_text("x" * 10_000_000)   # 10 MB
    from rfnry_rag.reasoning.cli import _read_directory_as_text, _MAX_DIR_READ_BYTES
    with pytest.raises(ValueError, match="exceeds"):
        _read_directory_as_text(tmp_path)
```

**Step 2 — Red.**

**Step 3 — Implementation:**

```python
_MAX_DIR_READ_BYTES = 5_000_000   # mirror _MAX_INGEST_CHARS in the retrieval SDK

def _read_directory_as_text(path: Path) -> str:
    parts: list[str] = []
    total = 0
    for f in sorted(path.iterdir()):
        if not f.is_file():
            continue
        data = f.read_text()
        total += len(data)
        if total > _MAX_DIR_READ_BYTES:
            raise ValueError(f"directory aggregate read exceeds {_MAX_DIR_READ_BYTES} bytes")
        parts.append(f"[{f.name}]\n{data}")
    return "\n\n".join(parts)
```

(Refactor the inline loop into this helper if it isn't already factored.)

**Step 4 — Green.**

**Step 5 — Commit:**
```bash
git commit -m "security: cap aggregate size of reasoning CLI directory reads"
```

---

## Closeout

After every task group lands:

```bash
uv run poe check
uv run poe typecheck
uv run poe test
git log --oneline -20
```

Update `CHANGELOG.md` with a "Cleanup & hardening" section covering the LOW fixes batch.

```bash
git commit -m "docs: CHANGELOG for 2026-04-23 round-2 low-priority cleanup"
```
