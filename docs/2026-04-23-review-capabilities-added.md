# Capabilities Added — 2026-04-23 Review

Minimal, additive summary of the new features and capabilities introduced while resolving the 45 findings from the 2026-04-23 comprehensive review. 25 commits across P0 (5), P1 (14), and M (6 grouped commits).

## New configuration surface

| Knob | Default | What it does |
|---|---|---|
| `LanguageModelClient.timeout_seconds` | `60` | Per-LLM-call timeout. Previously no timeout — a single stalled call could block indefinitely. |
| `Neo4jGraphStore.max_connection_pool_size` | `10` (was `100`) | Sized for a single-process SDK. |
| `Neo4jGraphStore.connection_acquisition_timeout` | `5.0s` (was `60.0s`) | Pool-exhaustion wait. |
| `Neo4jGraphStore.connection_timeout` | `5.0s` | New — TCP connect timeout. |
| `Neo4jGraphStore.password` | required | No default. Rejects empty at construction. |
| `QdrantVectorStore.scroll_timeout` | `30s` | Per-op timeout for scroll. |
| `QdrantVectorStore.write_timeout` | `30s` | Per-op timeout for upsert/delete. |
| `QdrantVectorStore.max_scroll_limit` | `10_000` | Clamp caller-supplied `limit` in scroll. |
| `SQLAlchemyMetadataStore.pool_timeout` | `10s` (was `30s` silent) | Surfaces pool exhaustion fast. |
| `PostgresDocumentStore.pool_timeout` | `10s` | Same. |
| `IngestionConfig.chunk_context_headers` | `True` | Renamed from `contextual_chunking` — old name still accepted with `DeprecationWarning`. |
| `BatchConfig.concurrency` | `5` (cap `20`) | Upper-bound enforced. |

## New validation

- `IngestionConfig.dpi`: `72 ≤ dpi ≤ 600` (was positive only)
- `RetrievalConfig.top_k`: `1 ≤ top_k ≤ 200`
- `RetrievalConfig.bm25_max_chunks`: `≤ 200_000`
- `GenerationConfig`: reject `grounding_enabled=True` with `threshold=0.0` (no-op)
- Cross-config: `tree_indexing.max_tokens_per_node ≤ tree_search.max_context_tokens`
- `IngestionConfig` / `RetrievalConfig` upper bounds previously unbounded — pathological values now fail at construction, not at runtime.

## New safety guarantees

- **Input bounds at public entry points**: query ≤ 32 000 chars, ingest content ≤ 5 000 000 chars, metadata ≤ 50 keys × 8 000-char values. Rejected with `ValueError` before any provider call.
- **Path-component whitelist** on `FilesystemDocumentStore`: `knowledge_id` and `source_type` must match `[A-Za-z0-9_-][A-Za-z0-9_-.]{0,127}`; resolved-path containment asserted; symlinks skipped during traversal.
- **XML/L5X hardened parser**: `resolve_entities=False`, `no_network=True`, `load_dtd=False`, `huge_tree=False` on every `etree.parse`/`iterparse` call.
- **Credential redaction**: `LanguageModelProvider.api_key`, `Neo4jGraphStore.password`/`username`, `LanguageModelClient.boundary_api_key` marked `repr=False`. Tracebacks never leak them.
- **Required ingestion methods** (`BaseIngestionMethod.required`): vector/document default `True`; failures raise `IngestionError` and skip the metadata commit so a source is never marked successful without its vectors. Graph/tree default `False` — their failures are logged and ingestion proceeds.
- **Rollback on partial init**: `RagEngine.initialize()` calls `shutdown()` on any exception before re-raising, so stores that opened before the failure are torn down.
- **Engine shutdown guarantee**: when `async with RagEngine(config)` fails during `__aenter__`, cleanup now happens via the explicit rollback path (not via `__aexit__`, which doesn't fire in that case).

## New correctness guarantees

- **Multi-collection symmetry**: `_retrieval_by_collection` and `_ingestion_by_collection` are populated for every configured collection at `initialize()`. Unknown collection names raise `ValueError` instead of silently falling back to the default pipeline (which previously mixed data across collections).
- **BM25 cache fan-out**: `_on_source_removed` and `_on_ingestion_complete` iterate every scoped collection's VectorRetrieval, not just the default namespace.
- **RRF-merged structured + unstructured retrieval**: `_merge_retrieval_results` runs `reciprocal_rank_fusion` over both ranked lists instead of sorting by raw score. Previously structured cosine scores (0–1) always beat unstructured RRF scores (0.01–0.05) regardless of relevance.
- **Graceful degradation**: `_retrieve_chunks` uses `return_exceptions=True` and logs per-path failures; one crashed retrieval path no longer kills the whole query.
- **Retrieval-error surfacing**: when every query variant fails (e.g. Qdrant down during a multi-query rewrite), `retrieve()` now raises `RetrievalError` instead of returning `[]`, so callers can distinguish total failure from an empty match set.
- **Batch drain safety**: `BatchIngestionService` final gather uses `return_exceptions=True` so one failing in-flight task doesn't cancel peers or leave stats inconsistent.
- **Event-loop hygiene**: `AnalyzedIngestionService` hashes files via `asyncio.to_thread` (previously blocked the event loop on large PDFs).
- **`MethodNamespace` duplicate guard**: raises `ValueError` on duplicate method names instead of silently overwriting.

## New schema durability

- **`rag_schema_meta` version table**: `SQLAlchemyMetadataStore.initialize()` reads + advances a schema version inside a single transaction using `INSERT ... ON CONFLICT DO NOTHING` (Postgres) / `INSERT OR IGNORE` (SQLite). Concurrent process starts serialise on the version row; schema-version > code refuses the connection (downgrade-protection).

## New stream semantics

- **Terminal `StreamEvent` on error**: `generate_stream` emits `StreamEvent(type="done", grounded=False)` with the error message before re-raising `GenerationError`, so consumers always see a structured terminal marker rather than an unexpected exception bubbling out of the async generator.

## New privacy controls

- **`rfnry_rag.common.logging.query_logging_enabled()`** shared helper, consumed by `retrieval/search/service.py` and `step_back.py` (the latter previously logged raw queries at INFO, bypassing the gate). All query-text logs are now consistently gated behind `RFNRY_RAG_LOG_QUERIES=true`.
- **`reasoning` CLI `init`** now `chmod 0o600` on `config.toml`, matching the retrieval side.
- **Qdrant startup warning** when `url` is `http://` without an `api_key` (plaintext + no auth prod footgun).

## New / improved exports

Retrieval:
- `BaseChunkRefinement` (protocol) — for custom refiner types
- `BaseRetrievalJudgment` (protocol) — for custom judge types

Reasoning:
- `BaseEmbeddings` (re-export) — for custom embeddings integrations
- `BaseSemanticIndex` (re-export) — for custom index integrations

CLI (`rfnry_rag.common.cli`):
- `OutputMode`, `get_output_mode`, `get_api_key` — were duplicated across retrieval/reasoning CLIs; now shared.

## Public-API breaking changes (documented per-commit)

1. `rfnry_rag.common.errors.BaseException` → **renamed to `SdkBaseError`**. The old name shadowed the Python builtin. Not re-exported at top-level.
2. `Neo4jGraphStore(password=...)` is **required** — no default.
3. `contextual_chunking` is **deprecated** in favour of `chunk_context_headers` (aliased with `DeprecationWarning` for one release).
4. `BOUNDARY_API_KEY` collision **raises `ConfigurationError`** (was a warning).

## New reusable utilities

- `rfnry_rag.common.embeddings.embed_batched` — shared between retrieval and reasoning.
- `rfnry_rag.common.cli.get_api_key` — env-var lookup with CLI-friendly error.
- `rfnry_rag.common.logging.query_logging_enabled` — the gate.

## Deleted

- `retrieval/modules/retrieval/search/vector.py` (orphaned `VectorSearch`; duplicated by `methods/vector.py`).
- Stale path-banner comment in `retrieval/modules/retrieval/base.py`.

## Regression coverage added

689 total tests (was 629 at the start of the review). New coverage:

- XXE hardening (XML + L5X)
- Required ingestion methods (3 paths: required-fails, optional-fails, missing-attr)
- BaseException shadowing regression
- Engine init rollback
- Filesystem path-traversal rejection (parametrised, including symlinks)
- step_back query-log gate
- Neo4j `__repr__` redaction + empty-password rejection
- `LanguageModelProvider` `__repr__` redaction
- LLM timeout validation
- Qdrant per-op timeout knobs + plaintext-no-auth warning
- Config upper bounds (`dpi`, `top_k`, `bm25_max_chunks`)
- Schema version table (populate, downgrade refuse, idempotent re-init)
- RRF merge scale-free behaviour
- Multi-collection symmetric maps (populate, unknown-collection rejection, scoped reuse)
- BM25 cache fan-out
- `contextual_chunking` deprecation shim
