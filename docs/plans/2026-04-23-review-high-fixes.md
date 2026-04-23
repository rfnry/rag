# Comprehensive Review — High Fixes (P1)

> **For Claude:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task.

**Prereq:** `2026-04-23-review-critical-fixes.md` is merged. Several P1s depend on protocol changes or rename done there.

**Goal:** Resolve the 14 P1 findings from the 2026-04-23 comprehensive review. These are the difference between "works on the happy path" and "works under load, adversarial inputs, and transient failure."

**Architecture:** Grouped into four phases: **security** (P1.1, P1.2, P1.14), **operational safety** (P1.3–P1.6, P1.13), **correctness under concurrency** (P1.7, P1.8, P1.9, P1.10, P1.11), **documentation trap** (P1.12). Phases are independent; within a phase, one commit per task.

**Paths** are relative to `packages/python/` unless absolute. Every task ends with `uv run poe test && uv run poe check && uv run poe typecheck` green before commit.

---

## Phase A — Security

### Task A1 — Sanitize `knowledge_id` and `source_type` in `FilesystemDocumentStore`

**Finding P1.1.** `stores/document/filesystem.py:60-65, 212` uses `knowledge_id` and `source_type` directly as directory path components without sanitization. A caller passing `knowledge_id="../../../etc"` can write arbitrary paths relative to the store root.

**Files:**
- Modify: `src/rfnry_rag/retrieval/stores/document/filesystem.py`
- Test: `src/rfnry_rag/retrieval/tests/test_filesystem_document_store.py` (extend)

**Step 1 — Write the failing test:**

```python
# test_filesystem_document_store.py — append
import pytest
from rfnry_rag.retrieval.stores.document.filesystem import FilesystemDocumentStore


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_id", ["../etc", "../../outside", "/abs/path", "a/b/c", ".."])
async def test_rejects_traversal_in_knowledge_id(tmp_path, bad_id):
    store = FilesystemDocumentStore(base_path=tmp_path)
    await store.initialize()
    with pytest.raises(ValueError, match="invalid.*component"):
        await store.save_document(
            source_id="src_01", content="x", knowledge_id=bad_id, source_type="type"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_type", ["../etc", "cron.d", "a/b"])
async def test_rejects_traversal_in_source_type(tmp_path, bad_type):
    store = FilesystemDocumentStore(base_path=tmp_path)
    await store.initialize()
    with pytest.raises(ValueError, match="invalid.*component"):
        await store.save_document(
            source_id="src_01", content="x", knowledge_id="kb", source_type=bad_type
        )
```

**Step 2 — Add the validator:**

```python
# src/rfnry_rag/retrieval/stores/document/filesystem.py
import re

_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9_\-.]{1,128}$")

def _safe_path_component(value: str, *, field: str) -> str:
    if not _SAFE_COMPONENT.match(value) or value in {".", ".."}:
        raise ValueError(f"invalid {field} path component: {value!r}")
    return value
```

Apply to `knowledge_id` and `source_type` wherever they become path components (save + load). Keep `_DEFAULT_KB` and `_UNTYPED` fallbacks for `None` inputs.

**Step 3 — Belt-and-suspenders:** after path construction, assert containment:

```python
resolved = (self._base_path / kb_dir / st_dir).resolve()
if not resolved.is_relative_to(self._base_path.resolve()):
    raise ValueError("resolved path escapes base_path")
```

**Step 4 — Commit:**

```bash
git commit -m "security: reject path-traversal components in FilesystemDocumentStore

knowledge_id and source_type were written directly as directory names.
A caller passing '../../../etc' could write files outside base_path.
Whitelist-validate both against [A-Za-z0-9_-.]{1,128} and assert
resolved containment."
```

---

### Task A2 — Gate step-back query logging behind `RFNRY_RAG_LOG_QUERIES`

**Finding P1.2.** `retrieval/modules/retrieval/search/rewriting/step_back.py:32` logs `query[:60]` unconditionally at INFO, bypassing the gate applied at `search/service.py:49-52`.

**Files:**
- Modify: `src/rfnry_rag/retrieval/modules/retrieval/search/rewriting/step_back.py`
- Test: `src/rfnry_rag/retrieval/tests/test_query_logging_gate.py` (extend)

**Step 1 — Extract the gate into a shared helper** (`common/logging.py`):

```python
# src/rfnry_rag/common/logging.py
import os

def query_logging_enabled() -> bool:
    return os.environ.get("RFNRY_RAG_LOG_QUERIES", "").lower() == "true"
```

Refactor `search/service.py:49-52` to call this helper. No behavior change.

**Step 2 — Apply to step_back:**

```python
# src/rfnry_rag/retrieval/modules/retrieval/search/rewriting/step_back.py:32
# BEFORE:
logger.info("step-back: '%s' -> '%s'", query[:60], result.broader_query[:60])

# AFTER:
if query_logging_enabled():
    logger.info("step-back: '%s' -> '%s'", query[:60], result.broader_query[:60])
else:
    logger.info("step-back rewrite completed")
```

**Step 3 — Grep for other violations:**

```bash
grep -rn 'logger\.\(info\|warning\|error\|debug\)[^"]*".*%s.*query' src/rfnry_rag/retrieval/
```

Audit each hit. Apply the same gate to any that log raw query text.

**Step 4 — Extend tests:** add one in `test_query_logging_gate.py` that drives `step_back.rewrite(...)` with `RFNRY_RAG_LOG_QUERIES` unset and asserts the raw query is not in captured log output.

**Commit:**

```bash
git commit -m "security: gate step_back query logging behind RFNRY_RAG_LOG_QUERIES

step_back.py unconditionally logged the first 60 chars of the user
query at INFO, bypassing the privacy gate applied elsewhere. Extract
query_logging_enabled() helper and apply it here too."
```

---

### Task A3 — Redact credentials in Neo4j store `__repr__`

**Finding P1.14.** `stores/graph/neo4j.py:155-166` is a `@dataclass` with `password: str = "password"` (Neo4j default). `__repr__` includes every field by default, so any exception traceback leaks the password. `LanguageModelProvider` has the same smell for `api_key`.

**Files:**
- Modify: `src/rfnry_rag/retrieval/stores/graph/neo4j.py`
- Modify: `src/rfnry_rag/common/language_model.py`
- Test: `src/rfnry_rag/retrieval/tests/test_store_credential_safety.py` (extend)

**Step 1 — Test:**

```python
# test_store_credential_safety.py — append
from rfnry_rag.retrieval.stores.graph.neo4j import Neo4jGraphStore
from rfnry_rag.common.language_model import LanguageModelProvider

def test_neo4j_repr_does_not_leak_password():
    store = Neo4jGraphStore(uri="neo4j://x", username="u", password="TOPSECRET")
    assert "TOPSECRET" not in repr(store)

def test_language_model_provider_repr_does_not_leak_api_key():
    p = LanguageModelProvider(provider="openai", model="m", api_key="sk-TOPSECRET")
    assert "TOPSECRET" not in repr(p)
```

**Step 2 — Fix:**

```python
# neo4j.py
from dataclasses import field

@dataclass
class Neo4jGraphStore:
    uri: str
    username: str = field(default="neo4j", repr=False)
    password: str = field(default="", repr=False)  # default "" + __post_init__ check
    ...

    def __post_init__(self) -> None:
        if not self.password:
            raise ConfigurationError("Neo4jGraphStore requires password")
```

Same `field(repr=False)` treatment for `api_key` in `LanguageModelProvider`.

**Commit:**

```bash
git commit -m "security: redact password/api_key from dataclass __repr__

Tracebacks and log lines that include the repr of Neo4jGraphStore or
LanguageModelProvider were leaking credentials. Mark the sensitive
fields repr=False and reject empty Neo4j passwords at construction
(the Neo4j default 'password' was a landmine)."
```

---

## Phase B — Operational safety

### Task B1 — Add explicit LLM call timeout

**Finding P1.3.** `LanguageModelClient` has no `timeout` field. With BAML's 3-retry loop and no per-call timeout, a single hung LLM call (rate-limit stall, network partition) can block the event loop indefinitely.

**Files:**
- Modify: `src/rfnry_rag/common/language_model.py`
- Modify: BAML client options builder (`_build_client_options`)
- Test: `src/rfnry_rag/common/tests/test_build_registry.py` (extend)

**Step 1 — Test:**

```python
def test_language_model_client_default_timeout():
    client = LanguageModelClient(provider=some_provider, model="m")
    assert client.timeout_seconds == 60

def test_language_model_client_rejects_non_positive_timeout():
    with pytest.raises(ConfigurationError):
        LanguageModelClient(provider=some_provider, model="m", timeout_seconds=0)
```

**Step 2 — Add the field:**

```python
# common/language_model.py
@dataclass
class LanguageModelClient:
    ...
    timeout_seconds: int = 60

    def __post_init__(self) -> None:
        ...
        if self.timeout_seconds <= 0:
            raise ConfigurationError("timeout_seconds must be positive")
```

Thread it into `_build_client_options` and pass to BAML's `add_llm_client` options. Check BAML-py docs for the exact option key (`request_timeout`, `timeout`, or provider-specific).

**Commit:**

```bash
git commit -m "feat: add per-call timeout to LanguageModelClient

Default 60s. Guards against indefinite hangs on LLM calls that
previously relied solely on BAML's retry-without-timeout loop."
```

---

### Task B2 — Tighten Neo4j pool sizing and acquisition timeout

**Finding P1.4.** `stores/graph/neo4j.py:165` uses `connection_acquisition_timeout=60` (locks an event-loop slot for 60s on pool exhaustion) and `max_connection_pool_size=100` (sized for a server, not an SDK).

**Files:**
- Modify: `src/rfnry_rag/retrieval/stores/graph/neo4j.py`
- Test: `src/rfnry_rag/retrieval/tests/test_neo4j_graph_store.py` (extend)

**Step 1 — Change defaults and expose them:**

```python
@dataclass
class Neo4jGraphStore:
    ...
    max_connection_pool_size: int = 10
    connection_acquisition_timeout: float = 5.0
    connection_timeout: float = 5.0
```

Pass them explicitly when constructing the driver.

**Step 2 — Test:** assert that the driver is constructed with the new defaults (inspect the mock call args in the existing test).

**Commit:**

```bash
git commit -m "config: tighten Neo4j pool (size 10, acquisition timeout 5s)

Previous defaults (size 100, timeout 60s) were copied from Neo4j
server deployment guides — inappropriate for a single-process SDK.
Expose all three timeouts as dataclass fields for operator tuning."
```

---

### Task B3 — Add per-operation timeouts to Qdrant store

**Finding P1.5.** `stores/vector/qdrant.py` sets an HTTP-client-level `timeout=10` but scroll/delete/retrieve/count take no per-call timeout and have no upper bound on `limit`.

**Files:**
- Modify: `src/rfnry_rag/retrieval/stores/vector/qdrant.py`
- Test: `src/rfnry_rag/retrieval/tests/test_store_pool_knobs.py` (extend)

**Step 1 — Expose scroll/write timeouts:**

```python
@dataclass
class QdrantVectorStore:
    url: str = "http://localhost:6333"
    api_key: str | None = None
    timeout: int = 10         # existing — short-op default
    scroll_timeout: int = 30
    write_timeout: int = 30
    max_scroll_limit: int = 10_000
```

Pass `timeout=self.scroll_timeout` to `scroll` calls, `timeout=self.write_timeout` to `upsert` / `delete`. Clamp caller-supplied `limit` in `scroll` to `max_scroll_limit`.

**Step 2 — Warn on plaintext-no-auth:**

```python
# __post_init__ or initialize():
if self.url.startswith("http://") and not self.api_key:
    logger.warning("qdrant: plaintext HTTP with no API key — do not use in production")
```

**Commit:**

```bash
git commit -m "config: add per-operation timeouts and scroll limit to QdrantVectorStore

The client-level timeout only covers the HTTP handshake, not slow
scroll/upsert operations. Add explicit scroll_timeout and
write_timeout (default 30s each), clamp scroll limit to 10k, and
warn when http:// is used without an API key."
```

---

### Task B4 — Upper-bound validation on `dpi`, `top_k`, `bm25_max_chunks`

**Finding P1.6.** Current `__post_init__` only enforces positivity. `dpi=10000` or `top_k=100000` causes OOM, not slowness.

**Files:**
- Modify: `src/rfnry_rag/retrieval/server.py` (three `__post_init__` methods)
- Test: `src/rfnry_rag/retrieval/tests/test_config_validation.py` (extend)

**Step 1 — Tests:**

```python
@pytest.mark.parametrize("bad_dpi", [71, 601, 10000])
def test_ingestion_config_rejects_out_of_range_dpi(bad_dpi):
    with pytest.raises(ConfigurationError):
        IngestionConfig(dpi=bad_dpi)

@pytest.mark.parametrize("bad_k", [201, 10_000])
def test_retrieval_config_rejects_huge_top_k(bad_k):
    with pytest.raises(ConfigurationError):
        RetrievalConfig(top_k=bad_k)

def test_retrieval_config_rejects_huge_bm25_max_chunks():
    with pytest.raises(ConfigurationError):
        RetrievalConfig(bm25_max_chunks=300_000)
```

**Step 2 — Validators:**

```python
# IngestionConfig.__post_init__
if not (72 <= self.dpi <= 600):
    raise ConfigurationError("dpi must be between 72 and 600")

# RetrievalConfig.__post_init__
if self.top_k > 200:
    raise ConfigurationError("top_k must be <= 200")
if self.bm25_max_chunks > 200_000:
    raise ConfigurationError(
        "bm25_max_chunks > 200_000 risks OOM; use sparse_embeddings instead"
    )
```

**Commit:**

```bash
git commit -m "config: bound dpi/top_k/bm25_max_chunks to safe ranges

OOM-class values (dpi=10000, top_k=100000, bm25_max_chunks=500k) now
raise ConfigurationError at construction instead of crashing at
runtime. Bounds: dpi 72-600, top_k <=200, bm25_max_chunks <=200k."
```

---

### Task B5 — Add a migration version table to `SQLAlchemyMetadataStore`

**Finding P1.13.** `stores/metadata/sqlalchemy.py:118-141` runs `ALTER TABLE ... ADD COLUMN` with no version table. Two processes racing on first boot can double-apply; the migration has no audit trail.

**Files:**
- Modify: `src/rfnry_rag/retrieval/stores/metadata/sqlalchemy.py`
- Test: `src/rfnry_rag/retrieval/tests/test_metadata_store_migration.py` (extend)

**Step 1 — Introduce a version table:**

```python
# sqlalchemy.py
_SCHEMA_VERSION = 1  # bump on every additive schema change

class _SchemaMeta(Base):
    __tablename__ = "rag_schema_meta"
    key: Mapped[str] = mapped_column(String, primary_key=True)
    value: Mapped[int] = mapped_column(Integer, nullable=False)
```

In `initialize()`, after creating base tables, run:
- Select `value` where `key='schema_version'`.
- If missing or lower than `_SCHEMA_VERSION`, run migrations and write the new version **inside the same transaction**.
- If higher, raise (downgrade is not supported).

Idempotency: use `INSERT ... ON CONFLICT DO NOTHING` (Postgres) / `INSERT OR IGNORE` (SQLite) to initialize the row, then `UPDATE` to advance.

**Step 2 — Tests:** concurrent-start test (two `asyncio.gather` initializes on the same DB file, both succeed, migration runs exactly once).

**Commit:**

```bash
git commit -m "feat: add rag_schema_meta version table for idempotent migrations

Prevents double-application of ALTER TABLE ADD COLUMN when two
processes initialize the same DB concurrently, and creates an audit
trail. Additive migrations only; downgrade raises."
```

---

## Phase C — Correctness under concurrency

### Task C1 — Fix score-scale mismatch in `_merge_retrieval_results`

**Finding P1.7.** `server.py:1033-1050` merges RRF-fused unstructured scores (~0.01–0.05) with cosine structured scores (0–1) and sorts by raw score. Structured results always win regardless of relevance.

**Fix:** run a secondary RRF pass over the two ranked lists instead of comparing raw scores.

**Files:**
- Modify: `src/rfnry_rag/retrieval/server.py:1033-1050`
- Test: `src/rfnry_rag/retrieval/tests/test_fusion.py` (extend)

**Step 1 — Test:** craft unstructured with RRF score 0.04 ranked #1 and structured with cosine 0.3 ranked #1; after merge, the top result must not be solely determined by raw score.

**Step 2 — Reuse existing `reciprocal_rank_fusion` helper from `retrieval/modules/retrieval/search/fusion.py`:**

```python
@staticmethod
def _merge_retrieval_results(
    unstructured: list[RetrievedChunk],
    structured: list[RetrievedChunk],
) -> list[RetrievedChunk]:
    if not structured:
        return unstructured
    if not unstructured:
        return structured
    from rfnry_rag.retrieval.modules.retrieval.search.fusion import reciprocal_rank_fusion
    return reciprocal_rank_fusion([unstructured, structured], weights=[1.0, 1.0])
```

Keep dedup-by-chunk-id (RRF should handle it, but verify).

**Commit:**

```bash
git commit -m "fix: RRF-merge unstructured and structured retrieval instead of raw-score sort

Before: structured cosine scores (0-1) always beat unstructured RRF
scores (0.01-0.05) regardless of relevance. Now both ranked lists go
through reciprocal_rank_fusion for a scale-free merge."
```

---

### Task C2 — Fix multi-collection retrieval fallback to default pipeline

**Finding P1.8.** `server.py:504-519` populates `_retrieval_by_collection` for non-default collections but leaves `_ingestion_by_collection` empty. A retrieve call to a non-default collection that was never explicitly ingested to silently falls back to the default pipeline.

**Files:**
- Modify: `src/rfnry_rag/retrieval/server.py`
- Test: `src/rfnry_rag/retrieval/tests/test_engine_multi_collection.py` (new)

**Step 1 — Test:** set up a vector store with `collections=["a", "b"]`, initialize, then retrieve against `"b"` without any prior ingest — assert results come from collection `b`, not `a`.

**Step 2 — Populate both maps symmetrically in `initialize()`:**

Rewrite the collection loop so that for every collection (including the first), an entry is placed in `_retrieval_by_collection` and `_ingestion_by_collection`. The first collection uses the already-constructed default services; the rest use `scoped(...)`.

Also fix `_get_retrieval` to raise (not silently fall back) when `collection` is specified but not in the map.

**Commit:**

```bash
git commit -m "fix: populate per-collection retrieval and ingestion maps symmetrically

Previously a retrieve() call on a non-default collection that was
never ingested to returned results from the default collection
silently, mixing data across collections. Both maps now get an
entry for every known collection at initialize() time."
```

---

### Task C3 — Invalidate BM25 cache on all scoped collections

**Finding P1.9.** `_on_source_removed` and `_on_ingestion_complete` only touch `self._retrieval_namespace.vector` (default collection). Scoped collections keep stale BM25 indexes.

**Files:** `src/rfnry_rag/retrieval/server.py:823-835`

Iterate all entries in `_retrieval_by_collection`:

```python
async def _on_source_removed(self, knowledge_id: str | None) -> None:
    for retrieval_service, _ in self._retrieval_by_collection.values():
        for method in retrieval_service._retrieval_methods:
            if method.name == "vector" and hasattr(method, "invalidate_cache"):
                await method.invalidate_cache(knowledge_id)
```

If `_retrieval_by_collection` is empty (single-collection setup), fall back to `_retrieval_namespace`.

**Commit:**

```bash
git commit -m "fix: invalidate BM25 cache on all scoped collections, not just default

After remove_source/ingest on a non-default collection, stale BM25
results kept being returned until engine restart."
```

---

### Task C4 — Hash file off the event loop in `AnalyzedIngestionService`

**Finding P1.10.** `ingestion/analyze/service.py:95` calls sync `compute_file_hash(file_path)` directly. `chunk/service.py:148` does it right with `asyncio.to_thread`.

**Files:** `src/rfnry_rag/retrieval/modules/ingestion/analyze/service.py`

```python
file_hash_value = await asyncio.to_thread(compute_file_hash, file_path)
```

**Commit:**

```bash
git commit -m "fix: hash file off the event loop in AnalyzedIngestionService

chunk/service.py:148 already uses asyncio.to_thread; analyze path
was still blocking the event loop during file I/O."
```

---

### Task C5 — Don't swallow errors in batch drain `gather`

**Finding P1.11.** `ingestion/chunk/batch.py:165-166` does `await asyncio.gather(*in_progress)` without `return_exceptions=True`. The eager loop at `:148` handles it correctly — the drain should too.

**Files:** `src/rfnry_rag/retrieval/modules/ingestion/chunk/batch.py`

```python
results = await asyncio.gather(*in_progress, return_exceptions=True)
for r in results:
    if isinstance(r, BaseException):
        logger.exception("batch ingestion task failed", exc_info=r)
        stats.failed += 1
```

**Commit:**

```bash
git commit -m "fix: consistently handle exceptions in BatchIngestionService drain

The final gather was not using return_exceptions=True, so a single
in-flight failure propagated as the return value and cancelled the
rest, leaving stats inconsistent. Match the eager-loop behavior at
the head of the stream."
```

---

## Phase D — Documentation trap

### Task D1 — Rename `contextual_chunking` to reflect what it actually does

**Finding P1.12.** The name implies Anthropic-style LLM-generated context per chunk (which *does* call an API). The implementation at `chunk/context.py:22-48` is pure string templating. If the real technique is ever added, existing users silently start paying per-chunk LLM calls.

**Files:**
- Modify: `src/rfnry_rag/retrieval/server.py:76` — rename `contextual_chunking` → `chunk_context_headers`
- Modify: `src/rfnry_rag/retrieval/modules/ingestion/chunk/context.py` + call sites
- Modify: `src/rfnry_rag/retrieval/cli/config.py` (TOML key mapping)
- Add: deprecation shim — accept the old name with a warning for one release, then remove

**Step 1 — Rename the field, keep a deprecation alias:**

```python
# server.py IngestionConfig
chunk_context_headers: bool = True
# deprecated — remove after one release
contextual_chunking: bool | None = None

def __post_init__(self) -> None:
    if self.contextual_chunking is not None:
        import warnings
        warnings.warn(
            "contextual_chunking is deprecated; use chunk_context_headers",
            DeprecationWarning, stacklevel=2,
        )
        self.chunk_context_headers = self.contextual_chunking
    ...
```

**Step 2 — Rename in the CLI TOML loader** and in `CLAUDE.md` if mentioned.

**Commit:**

```bash
git commit -m "refactor: rename contextual_chunking -> chunk_context_headers

The flag enables string-templated context headers, not the
LLM-based contextual chunking technique the name implies. Renamed
to reflect actual behavior; old name accepted with DeprecationWarning
for one release."
```

---

## Final verification

```bash
uv run poe test && uv run poe check && uv run poe typecheck
git log --oneline -20
```

14 focused commits across 4 phases. Open a single PR per phase to keep reviewer load manageable.
