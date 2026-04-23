# Comprehensive Review — Medium + Low Fixes

> **For Claude:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task.

**Prereq:** P0 (`2026-04-23-review-critical-fixes.md`) and ideally P1 (`2026-04-23-review-high-fixes.md`) merged. Several tasks here assume renames and helpers introduced there.

**Goal:** Close out the tech-debt and hardening items from the 2026-04-23 comprehensive review. None of these is a shipping blocker on its own, but the aggregate is what separates "fragile under evolution" from "stable to ship against."

**Architecture:** Grouped by type so related files can be touched together:
- **M1** — Hardening (validators, pool knobs, warnings)
- **M2** — Refactor (dead code, duplication, public API)
- **M3** — Low-severity polish

Each task is still one commit. The TDD discipline from the P0/P1 plans applies, but some items are documentation-only — the commit message explains the rationale.

**Paths** are relative to `packages/python/` unless absolute.

---

## M1 — Hardening

### M1.1 — Add `pool_timeout=10` to Postgres engines

**Finding.** `stores/metadata/sqlalchemy.py:91-94` and `stores/document/postgres.py:57-60` do not set `pool_timeout`. SQLAlchemy's 30s default means a pool-exhausted call stalls 30s before raising.

**Patch:**

```python
# both files
kwargs: dict[str, Any] = {"pool_pre_ping": True, "pool_recycle": 1800, "pool_timeout": 10}
```

Expose as a constructor param for future tuning. Extend `test_store_pool_knobs.py` to assert the value reaches `create_async_engine`.

**Commit:** `config: set pool_timeout=10s on Postgres engines to surface exhaustion fast`

---

### M1.2 — Size limits on user-supplied inputs

**Finding P1/M1.** `query()`, `retrieve()`, `ingest_text()`, `metadata` dict — all unbounded. Monetary DoS + OOM vector.

**Constants** (`retrieval/server.py` module-level):

```python
MAX_QUERY_CHARS = 32_000
MAX_INGEST_CHARS = 5_000_000
MAX_METADATA_KEYS = 50
MAX_METADATA_VALUE_CHARS = 8_000
```

Validate at the public method entry points:

```python
def _validate_query_text(text: str) -> None:
    if len(text) > MAX_QUERY_CHARS:
        raise ValueError(f"query exceeds {MAX_QUERY_CHARS} chars")

def _validate_metadata(metadata: dict[str, Any] | None) -> None:
    if not metadata:
        return
    if len(metadata) > MAX_METADATA_KEYS:
        raise ValueError(f"metadata exceeds {MAX_METADATA_KEYS} keys")
    for k, v in metadata.items():
        if isinstance(v, str) and len(v) > MAX_METADATA_VALUE_CHARS:
            raise ValueError(f"metadata[{k!r}] value exceeds {MAX_METADATA_VALUE_CHARS} chars")
```

Call at the start of `query`, `query_stream`, `retrieve`, `ingest`, `ingest_text`. Add one test per limit.

**Commit:** `feat: enforce size limits on query text, ingest content, and metadata dict`

---

### M1.3 — Guard `MethodNamespace` against duplicate method names

**Finding.** `modules/namespace.py:14` builds a dict keyed by `method.name`. Duplicate names silently overwrite.

**Patch:**

```python
# MethodNamespace.__init__
seen: set[str] = set()
for m in methods:
    if m.name in seen:
        raise ValueError(f"duplicate method name in namespace: {m.name!r}")
    seen.add(m.name)
self._methods: dict[str, T] = {m.name: m for m in methods}
```

Test: construct with two methods named `"vector"`, assert `ValueError`.

**Commit:** `fix: reject duplicate method names in MethodNamespace`

---

### M1.4 — Warn when `analyzed_methods` is empty in structured ingestion setup

**Finding.** `server.py:467` uses `m.name == "document"` string match. Empty list still instantiates `AnalyzedIngestionService`, which later silently skips document storage.

**Patch:**

```python
if persistence.metadata_store and persistence.vector_store and ingestion.embeddings:
    analyzed_methods = [m for m in ingestion_methods if m.name == "document"]
    if not analyzed_methods:
        logger.warning(
            "structured ingestion enabled but no DocumentIngestion configured — "
            "analyzed phase 3 will skip document storage"
        )
    self._structured_ingestion = AnalyzedIngestionService(...)
```

**Commit:** `chore: warn when structured ingestion has no document method available`

---

### M1.5 — `_retrieve_chunks` should degrade gracefully on one-path failure

**Finding.** `server.py:932-936` does `asyncio.gather(unstructured, structured)` without `return_exceptions=True`. Inner `_search_single_query` degrades gracefully; this is inconsistent.

**Patch:**

```python
results = await asyncio.gather(
    unstructured.retrieve(query=retrieval_query, knowledge_id=knowledge_id, **tree_kwargs),
    structured.retrieve(query=retrieval_query, knowledge_id=knowledge_id),
    return_exceptions=True,
)
unstructured_chunks = results[0] if not isinstance(results[0], BaseException) else []
structured_chunks = results[1] if not isinstance(results[1], BaseException) else []
if isinstance(results[0], BaseException):
    logger.warning("unstructured retrieval failed: %s", results[0])
if isinstance(results[1], BaseException):
    logger.warning("structured retrieval failed: %s", results[1])
if not unstructured_chunks and not structured_chunks:
    raise RetrievalError("all retrieval paths failed")
```

**Commit:** `fix: degrade gracefully when one retrieval path fails in _retrieve_chunks`

---

### M1.6 — Apply TOML key validation to reasoning CLI loader

**Finding.** Retrieval got `_validate_toml_keys` in commit a4619ce. Reasoning's `cli/config.py:111-121` still does a bare `tomllib.load(f)`.

**Patch:** Extract `_validate_toml_keys` + `_ALLOWED_TOP_KEYS` into `common/cli.py` (they already partially live there — `CONFIG_DIR`, `load_dotenv`). Both SDK loaders call the shared helper with their own allowlist.

**Commit:** `refactor: share TOML top-key validation between retrieval and reasoning CLIs`

---

### M1.7 — `reasoning init` must chmod config.toml to 0o600

**Finding.** `reasoning/cli/commands/init.py:32` creates `config.toml` without chmod. Retrieval `init.py:40` does it right.

**One-line patch:** add `CONFIG_FILE.chmod(0o600)` after `CONFIG_FILE.write_text(...)`.

Extend `test_init_config_permissions.py` (currently retrieval-only) to cover reasoning.

**Commit:** `security: chmod reasoning config.toml to 0o600, matching retrieval behavior`

---

### M1.8 — Cross-config tree constraint

**Finding.** `tree_indexing.max_tokens_per_node` can exceed `tree_search.max_context_tokens`. Silently yields empty tree search results.

**Patch:** add a cross-config check in `RagEngine._validate_config`:

```python
if cfg.tree_indexing.enabled and cfg.tree_search.enabled:
    if cfg.tree_indexing.max_tokens_per_node > cfg.tree_search.max_context_tokens:
        raise ConfigurationError(
            "tree_indexing.max_tokens_per_node cannot exceed "
            "tree_search.max_context_tokens (a single indexed node "
            "would not fit in the search context window)"
        )
```

**Commit:** `config: validate tree_indexing.max_tokens_per_node <= tree_search.max_context_tokens`

---

### M1.9 — Skip symlinks in `FilesystemDocumentStore` traversal

**Finding.** `rglob` follows symlinks. Combined with the Phase A path-component fix, this is still a concern if an attacker can plant symlinks under `base_path`.

**Patch:**

```python
for p in self._base_path.rglob("*.md"):
    if p.is_symlink():
        continue
    ...
```

Apply to both `_find_file` and `_load_entries`. Add a test that plants a symlink and asserts it's skipped.

**Commit:** `security: skip symlinks during FilesystemDocumentStore traversal`

---

### M1.10 — Warn instead of default for Neo4j password

Covered by **P1.14 (Task A3)** in the HIGH plan. Verify no residual `password="password"` default after that task lands. No separate commit needed.

---

### M1.11 — Grounding threshold boundaries

**Finding.** `grounding_threshold=0.0` with `grounding_enabled=True` silently disables grounding. `grounding_threshold=1.0` blocks every answer.

**Patch** in `GenerationConfig.__post_init__`:

```python
if self.grounding_enabled and self.grounding_threshold == 0.0:
    raise ConfigurationError(
        "grounding_enabled=True with grounding_threshold=0 is a no-op"
    )
if self.grounding_enabled and self.grounding_threshold >= 1.0:
    logger.warning(
        "grounding_threshold=%.2f blocks virtually every answer", self.grounding_threshold
    )
```

**Commit:** `config: reject grounding_enabled with threshold=0; warn near 1.0`

---

### M1.12 — Upper ceiling on `BatchConfig.concurrency`

**Finding.** `batch.py:70` accepts `>=1` but no ceiling.

**Patch:**

```python
if self.concurrency > 20:
    raise ValueError("concurrency > 20 risks overwhelming the vector store")
```

**Commit:** `config: cap BatchConfig.concurrency at 20`

---

## M2 — Refactor

### M2.1 — Delete orphaned `search/vector.py`

**Finding.** `retrieval/modules/retrieval/search/vector.py` defines `VectorSearch` with no importers outside itself. Logic is duplicated inside `methods/vector.py` (`VectorRetrieval`).

**Patch:**

```bash
rm packages/python/src/rfnry_rag/retrieval/modules/retrieval/search/vector.py
```

Grep once more before deletion: `grep -rn "from.*search.vector import\|from.*search\.vector import\|import.*search\.vector" packages/python/`. If any hit, re-route it to `methods.vector.VectorRetrieval` first.

**Commit:** `chore: delete orphaned search/vector.py (VectorSearch duplicated by VectorRetrieval)`

---

### M2.2 — Consolidate `_embed_batched` into a shared helper

**Finding.** `reasoning/modules/clustering/service._embed_batched` duplicates `retrieval/modules/ingestion/embeddings/utils.embed_batched`.

**Patch:** move `embed_batched` to `rfnry_rag/common/embeddings.py` (new file) or to `common/concurrency.py` if it stays tiny. Update both importers. Delete the duplicate.

**Commit:** `refactor: share embed_batched between retrieval and reasoning`

---

### M2.3 — Move `OutputMode`, `get_output_mode`, `_get_api_key` into shared common/cli.py

**Finding.** Copy-pasted between `retrieval/cli/output.py` + `config.py` and their `reasoning/cli/` mirrors.

**Patch:** put all three in `rfnry_rag/common/cli.py`. Re-export from each SDK's `cli/constants.py` (or just import directly).

**Commit:** `refactor: move OutputMode/get_output_mode/_get_api_key into common/cli.py`

---

### M2.4 — Add `retrieval/common/concurrency.py` re-export

**Finding.** Layering inconsistency — reasoning has `reasoning/common/concurrency.py`, retrieval imports directly from `rfnry_rag.common.concurrency`.

**Patch:** create `retrieval/common/concurrency.py` with `from rfnry_rag.common.concurrency import run_concurrent  # noqa: F401`. Update retrieval internal imports to go through the SDK layer.

**Commit:** `refactor: add retrieval/common/concurrency.py re-export for layering consistency`

---

### M2.5 — Public API exports

**Finding.**
- `retrieval/__init__.py` missing `BaseChunkRefinement`, `BaseRetrievalJudgment`
- `reasoning/__init__.py` missing `BaseEmbeddings`, `BaseSemanticIndex`
- Stale path comment in `retrieval/modules/retrieval/base.py:1`

**Patch:** add to each `__init__.py`'s explicit imports and `__all__`. Delete the stale banner comment.

Extend `test_public_api.py` to assert each symbol is importable from the top-level SDK module.

**Commit:** `chore: export protocol base classes from public SDK API`

---

### M2.6 — Clarify `StructuredRetrieval` vs `"enrich"` naming

**Finding.** Class is `StructuredRetrieval`, `.name` is `"enrich"`. Confusing for future readers.

**Patch:** add one docstring line on the class explaining the historical rename. No code change.

**Commit:** `docs: note StructuredRetrieval / "enrich" naming asymmetry`

---

## M3 — Low polish

### M3.1 — Emit terminal `StreamEvent(type="error")` before raising inside `generate_stream`

**Finding.** `generation/service.py:207-208` raises inside an async generator without yielding a terminal event.

**Patch:**

```python
async def generate_stream(self, ...) -> AsyncIterator[StreamEvent]:
    try:
        ...
        async for token in ...:
            yield StreamEvent(type="token", content=token)
        yield StreamEvent(type="done")
    except Exception as exc:
        yield StreamEvent(type="error", content=str(exc))
        raise
```

Tests: assert that a failing stream emits one `"error"` event before the exception surfaces.

**Commit:** `fix: emit terminal error StreamEvent before raising in generate_stream`

---

### M3.2 — Early return from `check_embedding_migration` on empty model name

**Finding.** `migration.py:23` marks all sources stale when `embedding_model_name == ""` (retrieval-only mode). Log spam.

**Patch:**

```python
async def check_embedding_migration(metadata_store, embedding_model_name: str) -> int:
    if not embedding_model_name:
        return 0
    ...
```

**Commit:** `fix: skip stale-source check when no embedding model is configured`

---

### M3.3 — Use `async with` for Neo4j transactions

**Finding.** `neo4j.py:291-301` manual `begin_transaction` / `commit` / `rollback`. Success-path commit-server-error can leak the transaction.

**Patch:**

```python
async with await session.begin_transaction() as tx:
    await tx.run(...)
    # commit happens automatically on __aexit__ without exception
```

Apply to any other manual-transaction site. Grep: `grep -n "begin_transaction" src/rfnry_rag/retrieval/stores/graph/neo4j.py`.

**Commit:** `refactor: use async with for Neo4j transactions`

---

### M3.4 — Distinguish "all retrieval variants failed" from "no results"

**Finding.** `search/service.py:69-73` returns `[]` whether all variants crashed or simply found nothing.

**Patch:** track success count; if zero variants returned successfully (every one was a `BaseException`), raise `RetrievalError`. The existing empty-result case stays unchanged.

```python
successes = sum(1 for o in outcomes if not isinstance(o, BaseException))
if outcomes and successes == 0:
    raise RetrievalError("all retrieval variants failed")
```

**Commit:** `fix: raise RetrievalError when every query variant fails`

---

### M3.5 — `BOUNDARY_API_KEY` collision should error, not warn

**Finding.** `common/language_model.py:109-124` warns on collision; silent misconfiguration in multi-tenant setups.

**Patch:** change warning to `raise ConfigurationError(...)`. Update the single caller and extend `test_boundary_api_key.py`.

**Commit:** `fix: raise on BOUNDARY_API_KEY collision instead of warning`

---

### M3.6 — Replace hardcoded BAML fallback `gpt-4o-mini` with sentinel

**Finding.** `retrieval/baml/baml_src/clients.baml:36-43` uses a real model name as the fallback. Future deprecation landmine.

**Patch:** set model to `"UNCONFIGURED"` (or similar) so any unrouted call fails fast with a clear API error. Regenerate BAML clients: `poe baml:generate:retrieval`.

**Commit:** `chore: replace hardcoded gpt-4o-mini BAML fallback with UNCONFIGURED sentinel`

---

### M3.7 — Document `chunk_size` / `chunk_overlap` defaults

**Finding.** Values `500` / `50` are reasonable but uncommented.

**Patch:** add a one-line comment next to the dataclass defaults:

```python
chunk_size: int = 500    # ~100 words; fits typical 512-1536 token embedding windows
chunk_overlap: int = 50  # 10% overlap preserves boundary context
```

**Commit:** `docs: annotate chunk_size/chunk_overlap defaults`

---

### M3.8 — `collection` vs `knowledge_id` docstring

**Finding.** Easy to confuse for new readers. CLI default `collection="knowledge"` makes it worse.

**Patch:** add a class docstring on `PersistenceConfig` (or on `RagEngine.ingest`) that distinguishes the two:

> - `collection`: backend routing key (Qdrant collection name, Postgres schema)
> - `knowledge_id`: per-document partition filter applied at query time

**Commit:** `docs: clarify collection vs knowledge_id semantics`

---

## Final verification

```bash
uv run poe test && uv run poe check && uv run poe typecheck
git log --oneline -30
```

Expected: ~26 focused commits across M1 / M2 / M3. These can be batched into three PRs matching the groups, or merged incrementally — they have no cross-dependencies except where noted (M1.10 → Task A3, M1.6 depends on shared helper landing).
