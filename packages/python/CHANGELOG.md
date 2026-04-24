# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Resolves 45 findings from the 2026-04-23 comprehensive review across
correctness, security, operational safety, and hardening. 25 commits; 689
tests passing (up from 629).

### 2026-04-24 Round 4 Review

Cumulative fixes from the fourth-pass SDK audit. Primary achievement:
**four systematic contract-test sweeps** that now block future instances of
the same findings. 21 commits; test count 789 → 837.

**Contract-test sweeps (prevent future circling):**
- `test_baml_prompt_fence_contract.py` — every user-controlled BAML prompt parameter across retrieval + reasoning must be fenced. Sweep caught and fixed **35 unfenced parameters** across 24 functions in 13 BAML files (rounds 2-3 had only spot-fixed 7).
- `test_config_bounds_contract.py` — every int/float config field must have validation in `__post_init__` or a `# unbounded: <reason>` marker. Sweep fixed **8 unbounded knobs** (source_type_weights, temperature, num_variants, ClassificationConfig.concurrency, ClusteringConfig.{n_clusters,min_cluster_size,random_state}, ComplianceConfig.max_reference_length).
- Serial-await audit — 11 intentionally-serial loops annotated with `# SERIAL:` comments explaining why they aren't parallelised.
- `test_no_bare_valueerror_in_configs.py` — 33 bare `raise ValueError` replaced with new `ReasoningInputError(ReasoningError, ValueError)` across 7 reasoning modules.

**Ship blockers fixed:**
- `TreeRetrieval` dead code removed (never wired into MethodNamespace; round-3 parallelised an unreachable path) (A1).
- Lucene metacharacter escape on Neo4j fulltext queries — prevents wildcard-explosion DoS (A2/H1).
- `AnalyzedIngestionService._analyze_pdf` parallelised with bounded `asyncio.gather` (5 concurrent) — 200-page PDF no longer 200 serial LLM calls (A3/H2).
- `parent_chunk_overlap < parent_chunk_size` validator added — prevents infinite `_merge_splits` loop (A4/H4).
- `_migrate_missing_columns` verified idempotent via column introspection (A5/M1).
- Neo4j URI credentials scrubbed before startup logging (A6/M5).

**Performance:**
- Batch tree-index lookup eliminates N+1 DB query in `_run_tree_search` (C1/M6).
- `list_source_ids` lean projection — tree search no longer transfers megabyte `metadata_json`/`tree_index_json` payloads (C2/M7).
- `embed_batched` centralises sub-batch concurrency with `asyncio.Semaphore(3)`; removes competing provider-layer gather (C3/M8).
- BAML `ClientRegistry` cached in `GenerationService.__init__` instead of rebuilt per `generate()` call (D1/L7).
- `_synthesize_shared_entities` caps per-entity pairwise cross-reference expansion at 20 pages (D6).

**Security:**
- `TreeIndex.pages` count guard (≤ 100_000) added to `from_dict`, complementing round-3's depth/node guards on `TreeNode` (C4/F7).
- `compliance --references` CLI: `exists=True` + symlink/traversal containment check in `_read_directory_as_text` (D3/F5).
- Upper-bound pins on `lxml`, `fastembed`, `voyageai`, `neo4j`, `rank-bm25` (D4/F6).

**Safety / lifecycle:**
- `RagEngine.shutdown` idempotent (separate `_stores_opened` flag preserves the init-rollback path) (D5/L2).

**Docs:**
- CLAUDE.md test count, new config ceilings, collection-0 aliasing note (D2 / D7 / arch M10).

**Deferred from prior rounds:**
- Round-3 L8 (uniform scoped pipelines / remove index-0 special case) — comment extended (D7) but refactor still not pursued; tests codify the current behavior.

### 2026-04-24 Round 3 Review

Cumulative fixes from a third-pass SDK audit. 25 commits; test count 741 → 789.

**Round 2 regressions resolved:**
- `on_progress` callback now emits a final `(total, total)` completion event regardless of group composition (R2-A).
- `_dispatch_methods` TaskGroup `except*` handler now propagates `CancelledError`/`SystemExit`/`KeyboardInterrupt` unwrapped (R2-B).

**Performance:**
- BM25 index build scrolls chunks outside the cache lock — concurrent builds for different `knowledge_id`s no longer serialize (P-1).
- `TreeRetrieval.search` fans out per-source work with `asyncio.gather` (P-2).
- Embedding provider sub-batches run under a bounded `asyncio.Semaphore(3)` instead of serial `for` loop (P-3).
- Added DB index on `rag_sources.file_hash` with schema migration — `find_by_hash` no longer full-scans (P-4).
- `VectorIngestion` gathers dense + sparse embeddings concurrently (P-5).

**Config safety:**
- `RFNRY_RAG_BAML_LOG` validated against `{trace, debug, info, warn, error, off}` (C-1).
- All stores reject zero/negative timeout and pool knobs; SQLAlchemy accepts `-1` for `pool_recycle` (C-2).
- `TreeSearchConfig.max_sources_per_query` (`1–1000`, default 50) caps `_run_tree_search` fan-out (C-3).
- Upper bounds on tree indexing/search knobs: `toc_scan_pages ≤ 500`, `max_pages_per_node ≤ 200`, `max_tokens_per_node ≤ 200_000`, `max_steps ≤ 50`, `max_context_tokens ≤ 500_000` (C-4).
- `BatchConfig.batch_size ≤ 100_000` (C-5).
- `RetrievalConfig.bm25_max_indexes ∈ [1, 1000]` (L2).

**Security hardening:**
- Query-rewriting BAML prompts (HyDE, multi-query, step-back) fence `{{ query }}` in `======== QUERY START/END ========` (S-1).
- Reasoning BAML `AnalyzeContext` fences `{{ roles }}`; `CheckCompliance` fences `{{ dimensions }}` (S-2).
- Reasoning SDK configs cap `max_text_length ≤ 5_000_000` chars (S-3).
- `RagEngine.embed` and `embed_single` validate query length (L3).
- `TreeNode.from_dict` rejects tampered trees with depth > 100 or > 10 000 nodes (L4).
- `mapper._classify_relationship` returns `None` for unknown relationships instead of defaulting to `CONNECTS_TO`; unclassifiable cross-references are dropped (L5).

**Error typing:**
- `RuntimeError` → `ConfigurationError` for missing generation config at call time (three sites) (L6).
- New `InputError(RagError, ValueError)` for input-validation failures; back-compat with `except ValueError:` preserved (L7).

**Observability:**
- Neo4j and Qdrant stores log effective pool/timeout settings at startup (L1).

**Docs:**
- Clarified `RetrievalService.methods` is a concrete-class contract (L9).
- CLAUDE.md documents tree-search placement in the retrieval pipeline flow (L10).
- `Pipeline` docstring explains why `ClusteringService` is intentionally excluded (L11).

**Deferred:**
- L8 (uniform scoped pipelines) — deferred. The `collections[0]` special case is intentional per existing tests that codify first-collection-reuses-default-services behavior. Will require test-spec revision before it can land.

### 2026-04-23 Round 2 Review

Cumulative fixes from a second-pass SDK audit. 40 commits; test count 689 → 741.

**Correctness (critical & high):**
- Rejected `collection=` on structured ingestion paths instead of silently using the default collection (C1).
- Wired the previously-noop `on_progress` callback in `RagEngine.ingest` (C2).
- Fan out `_run_tree_search` across sources with `asyncio.gather` — serial loop removed (H1).
- Hardened Cypher relationship-type allowlist with a contract test and fixed a latent silent-fallback bug in `_validate_relation_type` (H2).
- Replaced private `_retrieval_methods` reach-through with a public `RetrievalService.methods` property (H3).
- Selected `DocumentIngestion` by `isinstance` rather than string-name filter (H4).

**Correctness (medium):**
- Scoped-collection ingestion now includes graph and tree methods (M1).
- Analyzed ingestion writes the document store exactly once in phase 3 (M2).
- `grounding_enabled` guard moved to `GenerationConfig.__post_init__` (M3).
- `tree_indexing.enabled` and `tree_search.enabled` without a `model` now fail at init (M4).
- Ingestion methods parallelise within required/optional groups via TaskGroup + gather (M5).

**Performance:**
- Voyage reranker uses `voyageai.AsyncClient` directly (M6).
- Embedding providers chunk batches at provider limits: OpenAI 2048, Voyage 128, Cohere 96 (M7).
- Metadata-store duplicate-hash check pushed into SQL WHERE (L12).

**Security hardening:**
- `GenerateAnswer` BAML prompt fences untrusted context to blunt prompt injection (M8).
- Postgres document search moved from f-string SQL assembly to SQLAlchemy Core expressions (M9).
- PyMuPDF upper-bounded; PDF parser guards size and page count (M10).
- `RagEngine.generate_step` validates query length (L13).
- Filesystem frontmatter values JSON-encoded to survive embedded delimiters (L14).
- Reasoning CLI directory reads capped at 5 MB aggregate (L15).

**Configuration surface:**
- Added `QdrantVectorStore(hybrid_prefetch_multiplier=...)` (L4).
- Added `PostgresDocumentStore(headline_max_words, headline_min_words, headline_max_fragments)` (L5).
- Added `RetrievalConfig(history_window=...)` (L6).
- `RFNRY_RAG_LOG_LEVEL` validates against known level names (L7).

**Observability:**
- Effective pool sizes and LLM client policy logged at startup (L8).

**Lifecycle / safety:**
- `run_concurrent` caller-supplied `concurrency` bounded to 1–100 (L9).
- `RagEngine.shutdown` tears down stores in reverse-init order and clears service refs (L10).
- BM25 cache initial read held under lock to prevent torn read vs invalidate (L11).

**Refactors & hygiene:**
- `IngestionService` internal kwarg renamed `contextual_chunking` → `chunk_context_headers` (L1).
- `MethodNamespace[T]` type parameter threaded through engine properties (L2).
- Shared synthesise-entities loop body extracted in analyzed ingestion (L3).
- `_escalation_result` loses unused `chunks=` parameter (M11).
- `_enabled_flows()` normalised to snake_case (M12).
- `reasoning/__init__.py::__all__` sorted; `RUF022` enabled (M13).
- `generate_step()` documented in retrieval README (M14).

### Breaking changes

- **`rfnry_rag.common.errors.BaseException` renamed to `SdkBaseError`**, and
  no longer re-exported at the top level of `rfnry_rag`. The old name
  shadowed the Python builtin: any code doing `from rfnry_rag import *`
  or `from rfnry_rag import BaseException` silently narrowed user
  `except BaseException:` clauses. Users should catch the specific
  subclasses (`RagError`, `ReasoningError`, `ConfigurationError`) or
  import `SdkBaseError` from `rfnry_rag.common.errors` directly.
- **`Neo4jGraphStore(password=...)` is now required.** The previous default
  `"password"` was the Neo4j community-edition default — the universally
  known credential was a landmine. Passing an empty password raises
  `ConfigurationError`. Neo4j pool defaults also tightened:
  `max_connection_pool_size=10` (was 100), `connection_acquisition_timeout=5.0s`
  (was 60.0s).
- **`IngestionConfig.contextual_chunking` deprecated; use
  `IngestionConfig.chunk_context_headers`.** The old name implied
  Anthropic-style LLM-generated per-chunk context; the implementation is
  pure string templating. The old name still works for one release but
  emits `DeprecationWarning`. CLI TOML: prefer `chunk_context_headers = true`.
- **`BOUNDARY_API_KEY` collision raises `ConfigurationError`** instead of
  warning. If two `LanguageModelClient` instances register different keys
  in the same process, the second raises — set `BOUNDARY_API_KEY` once at
  process start in multi-tenant scenarios.

### Added

- `LanguageModelClient.timeout_seconds` (default `60s`) — per-LLM-call
  timeout. Previously no timeout; a hung call could stall the event loop.
- `QdrantVectorStore.scroll_timeout` (default `30s`),
  `QdrantVectorStore.write_timeout` (default `30s`),
  `QdrantVectorStore.max_scroll_limit` (default `10_000`) — per-operation
  timeouts and scroll cap. Plaintext `http://` URL without an API key now
  emits a startup warning.
- `SQLAlchemyMetadataStore.pool_timeout` and
  `PostgresDocumentStore.pool_timeout` (default `10s`, was SQLAlchemy's
  silent 30s) — fail-fast on pool exhaustion.
- `Neo4jGraphStore.connection_timeout` (default `5.0s`) — explicit TCP
  connect timeout.
- `BaseIngestionMethod.required: bool` — vector/document default `True`,
  graph/tree default `False`. Failures in required methods raise
  `IngestionError` and skip the metadata commit so a source is never
  marked successful without its vectors.
- `rag_schema_meta` version table on `SQLAlchemyMetadataStore` — migrations
  are idempotent under concurrent process starts and refuse to operate
  against a schema newer than the code (downgrade protection).
- `ValueError` on `MethodNamespace` duplicate method names.
- `retrieval/common/concurrency.py` — re-export of `run_concurrent` for
  parity with `reasoning/common/concurrency.py`.
- `BaseChunkRefinement`, `BaseRetrievalJudgment` exported from
  `rfnry_rag.retrieval`; `BaseEmbeddings`, `BaseSemanticIndex` exported
  from `rfnry_rag.reasoning`.
- `rfnry_rag.common.embeddings.embed_batched` — shared between retrieval
  ingestion and reasoning clustering.
- `rfnry_rag.common.cli.OutputMode`, `get_output_mode`, `get_api_key` —
  shared CLI helpers (previously duplicated across retrieval/reasoning).
- `rfnry_rag.common.logging.query_logging_enabled` — gate helper for
  `RFNRY_RAG_LOG_QUERIES`.

### Changed

- `_merge_retrieval_results` now RRF-merges structured + unstructured
  results instead of raw-score sort. Previously structured cosine
  scores (0–1) always beat unstructured RRF scores (0.01–0.05)
  regardless of relevance.
- `RagEngine.initialize()` wraps setup in try/except and calls
  `shutdown()` on partial failure. Previously, an exception in
  mid-init leaked prior store connections (`__aexit__` doesn't fire
  when `__aenter__` raises).
- Multi-collection wiring: `_retrieval_by_collection` and
  `_ingestion_by_collection` populated symmetrically at `initialize()`.
  Unknown collection names raise `ValueError` instead of silently
  falling back to the default pipeline.
- `_on_source_removed` / `_on_ingestion_complete` fan out BM25-cache
  invalidation across every scoped collection, not just the default.
- `AnalyzedIngestionService` hashes files via `asyncio.to_thread`
  (previously blocked the event loop).
- `BatchIngestionService` drain uses `return_exceptions=True` so a
  single failing task doesn't cancel peers.
- `_retrieve_chunks` uses `return_exceptions=True`; one failing
  retrieval path logs + continues instead of killing the query.
- `search/service.retrieve()` raises `RetrievalError` when every query
  variant fails (was silently returning `[]`).
- `generate_stream` emits a terminal `StreamEvent(type="done",
  grounded=False)` before re-raising `GenerationError`.
- Neo4j `delete_by_source` uses `async with session.begin_transaction()`
  for guaranteed cleanup on commit-path server errors.
- BAML `Default` fallback client model set to sentinel
  `"UNCONFIGURED-route-through-build-registry"` so unrouted calls fail
  fast with a clear error instead of silently using `gpt-4o-mini`.

### Security

- **XXE hardening**: XML and L5X parsers use an explicit
  `etree.XMLParser(resolve_entities=False, no_network=True,
  load_dtd=False, huge_tree=False)` on every read.
- **Path-traversal rejection**: `FilesystemDocumentStore` validates
  `knowledge_id` and `source_type` against
  `^[A-Za-z0-9_-][A-Za-z0-9_-.]{0,127}$`, asserts resolved-path
  containment under `base_path`, and skips symlinks during `rglob`
  traversal.
- **Credential redaction**: `LanguageModelProvider.api_key`,
  `Neo4jGraphStore.password`/`username`, and
  `LanguageModelClient.boundary_api_key` marked `repr=False` so they
  never appear in tracebacks or log lines.
- **Query log gate**: the `step_back` rewriter previously logged raw
  query text at INFO bypassing the privacy gate. Now consistently
  gated behind `RFNRY_RAG_LOG_QUERIES=true` via the shared
  `query_logging_enabled()` helper.
- **`reasoning init` chmod 0o600** on generated `config.toml`, matching
  retrieval `init`.
- **Input size limits** at public entry points: query ≤ 32 000 chars,
  `ingest_text` content ≤ 5 000 000 chars, metadata ≤ 50 keys × 8 000
  chars.

### Fixed

- `RagEngine.vector_only(...)`, `.document_only(...)`, and `.hybrid(...)`
  presets now `initialize()` cleanly. Previously the grounding-gate
  guard fired whenever `grounding_threshold > 0` (default `0.5`)
  regardless of whether grounding was enabled, raising
  `ConfigurationError` on every retrieval-only preset.
- Required ingestion methods no longer silently commit a source when
  their upsert fails (vector upsert failure previously resulted in a
  valid-looking source with zero vectors in Qdrant).
- `check_embedding_migration` early-returns when no embedding model is
  configured so retrieval-only/document-only setups don't flood logs
  with stale-source warnings.

### Validation tightened

- `IngestionConfig.dpi`: `72 ≤ dpi ≤ 600`.
- `RetrievalConfig.top_k`: `≤ 200`.
- `RetrievalConfig.bm25_max_chunks`: `≤ 200 000`.
- `GenerationConfig`: reject `grounding_enabled=True` with
  `grounding_threshold=0.0` (no-op).
- Cross-config: `tree_indexing.max_tokens_per_node ≤
  tree_search.max_context_tokens`.
- `BatchConfig.concurrency`: cap at `20`.
- Reasoning CLI TOML: unknown top-level keys now raise `ConfigError`
  (matches retrieval's prior hardening).

### Removed

- `retrieval/modules/retrieval/search/vector.py` (`VectorSearch` class —
  orphaned; logic duplicated inside `methods/vector.py`).
- Stale `# src/rfnry-rag/retrieval/modules/retrieval/base.py` banner
  comment (referenced the old hyphenated package path).

---

## Prior history

See `git log --oneline` for commits predating the 2026-04-23 review.
