# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

rfnry-rag is a dual-SDK Python package providing two AI pipelines:
- **Retrieval SDK** (`src/rfnry-rag/retrieval/`) — Document ingestion, multi-path semantic search, LLM-grounded generation
- **Reasoning SDK** (`src/rfnry-rag/reasoning/`) — Text analysis, classification, clustering, compliance checking, evaluation, pipeline composition

Both SDKs share common infrastructure in `src/rfnry-rag/common/` (errors, language model config, logging, concurrency, CLI utilities). Each SDK has its own `common/` that re-exports from the shared common — never duplicate code between them.

## Commands

All tasks run via [poethepoet](https://github.com/nat-n/poethepoet). Prefix with `uv run` if not in the venv:

```bash
poe format                    # ruff format
poe check                     # ruff lint
poe check:fix                 # ruff lint with auto-fix
poe typecheck                 # mypy src/
poe test                      # pytest (asyncio_mode=auto, pythonpath=src)
poe test:cov                  # pytest with coverage
poe baml:generate:retrieval   # regenerate retrieval BAML clients
poe baml:generate:reasoning   # regenerate reasoning BAML clients
```

Run a single test: `pytest src/rfnry-rag/retrieval/tests/test_search.py::test_name -v`

## Architecture

### Package Structure

```
src/rfnry-rag/
├── __init__.py          # Re-exports everything from both SDKs
├── cli.py               # Unified CLI: rfnry-rag retrieval ... / rfnry-rag reasoning ...
├── common/              # Shared across both SDKs
│   ├── errors.py        # SdkBaseError, ConfigurationError (base classes)
│   ├── language_model.py # LanguageModelClient, LanguageModelProvider, build_registry (BAML ClientRegistry)
│   ├── logging.py       # get_logger (env: RFNRY_RAG_LOG_ENABLED, RFNRY_RAG_LOG_LEVEL)
│   ├── startup.py       # BAML version check (parameterized per SDK)
│   ├── concurrency.py   # run_concurrent helper
│   └── cli.py           # ConfigError, CONFIG_DIR, load_dotenv
├── retrieval/
│   ├── common/           # Re-exports from rfnry_rag.common + retrieval-specific (models, formatting, hashing, page_range)
│   ├── server.py         # RagEngine — main entry point, dynamic pipeline assembly
│   ├── modules/
│   │   ├── namespace.py  # MethodNamespace[T] — attribute access + iteration for pipeline methods
│   │   ├── ingestion/    # base.py (BaseIngestionMethod protocol), methods/ (vector, document, graph, tree), chunk/ (chunker, parsers, batch), analyze/ (analyzed 3-phase), embeddings/ (Embeddings facade), vision/ (Vision facade), tree/
│   │   ├── retrieval/    # base.py (BaseRetrievalMethod protocol), methods/ (vector, document, graph), search/ (service, fusion, reranking/ (Reranking facade), rewriting/ (BaseQueryRewriting)), refinement/ (BaseChunkRefinement), enrich/, judging (RetrievalJudgment), tree/
│   │   ├── generation/   # service, step, grounding, confidence
│   │   ├── knowledge/    # manager (CRUD), migration
│   │   └── evaluation/   # metrics (ExactMatch, F1, LLMJudgment), retrieval_metrics
│   ├── stores/           # vector/ (Qdrant), metadata/ (SQLAlchemy), document/ (Postgres, filesystem), graph/ (Neo4j)
                          # SQLAlchemyMetadataStore schema: rag_schema_meta (version/migration guard), rag_sources (source records + file_hash index), rag_page_analyses (per-page cache: composite key source_id+page_number, indexed page_hash, FK cascade-delete; Phase B1)
│   ├── cli/              # Click commands, config loader, output formatters
│   └── baml/             # baml_src/ (edit) + baml_client/ (generated, do not edit)
└── reasoning/
    ├── common/           # Re-exports from rfnry_rag.common
    ├── modules/
    │   ├── analysis/     # AnalysisService — intent, dimensions, entities, context tracking
    │   ├── classification/ # ClassificationService — LLM or hybrid kNN→LLM
    │   ├── clustering/   # ClusteringService — K-Means, HDBSCAN, LLM labeling
    │   ├── compliance/   # ComplianceService — policy violation checking
    │   ├── evaluation/   # EvaluationService — similarity + LLM judge scoring
    │   └── pipeline/     # Pipeline — sequential step composition
    ├── protocols.py      # BaseEmbeddings (from rfnry_rag.common.protocols), BaseSemanticIndex (structural typing)
    ├── cli/              # Click commands, config loader, output formatters
    └── baml/             # baml_src/ (edit) + baml_client/ (generated, do not edit)
```

### Entry Points

- **Retrieval:** `RagEngine` in `server.py` — async context manager. `async with RagEngine(config) as rag:`
- **Reasoning:** Services are standalone (`AnalysisService`, `ClassificationService`, etc.). `Pipeline` composes them sequentially.
- **CLI:** `rfnry-rag retrieval <cmd>` / `rfnry-rag reasoning <cmd>` (also standalone: `rfnry-rag-retrieval`, `rfnry-rag-reasoning`)
- **SDK import:** `from rfnry_rag import RagEngine, Pipeline, AnalysisService` — top-level re-exports everything from both SDKs

### Retrieval Pipeline Flow

The retrieval pipeline in `RagEngine` runs in this order:

1. **Query rewriting** (pre-retrieval, optional) — HyDE, multi-query, or step-back. Expands 1 query into multiple variants via an LLM call. Configured via `RetrievalConfig.query_rewriter`.
2. **Multi-path search** (per query) — pluggable retrieval methods run concurrently, results merged via reciprocal rank fusion with per-method weights:
   - **VectorRetrieval** — Dense similarity + SPLADE hybrid (if `sparse_embeddings`) + BM25 (if `bm25_enabled`), fused internally via RRF. Each method has `weight` and optional `top_k` override.
   - **DocumentRetrieval** — Full-text + substring search (requires document store)
   - **GraphRetrieval** — Entity lookup + N-hop traversal (requires graph store) — exposes `trace(entity_name, max_hops, relation_types, knowledge_id)` for programmatic N-hop queries returning `GraphPath` objects directly (vs `search()`'s RetrievedChunk conversion).
   - **Enrich** — Structured retrieval with field filtering (requires metadata store)
   - **Tree** — Runs when `TreeSearchConfig.enabled`; loads stored tree indexes for up to `max_sources_per_query` sources, runs an LLM-backed navigation per source, and injects results into the RRF fusion pool alongside the other retrieval paths. See `server.py::_run_tree_search` for orchestration; per-source work runs concurrently via `asyncio.gather`. Uses `list_source_ids` + `get_tree_indexes` batch queries to eliminate N+1 DB lookups. Requires metadata store. Note: the `TreeRetrieval` class was removed in round-4 (it was never wired into `MethodNamespace`); tree search runs exclusively via `_run_tree_search` in `server.py`.
3. **Reranking** (optional) — Cross-encoder reranking against original query (Cohere, Voyage)
4. **Chunk refinement** (optional) — Extractive (context window) or abstractive (LLM summarization) refinement
5. **Generation** (for `query()` only) — Grounding gate → LLM relevance gate → optional clarification → LLM generation. Context assembly via `chunks_to_context()` accepts `GenerationConfig.chunk_ordering` (`ChunkOrdering.SCORE_DESCENDING` default, `PRIMACY_RECENCY`, or `SANDWICH`) to mitigate the Lost-in-the-Middle U-shaped-attention effect (Liu et al., TACL 2024). Same ordering threads through `GenerationService` and `StepGenerationService` — no per-call-site knob (R4, 2026-04-25).

**Optional trace** (R8.1, 2026-04-27) — pass `trace=True` to `RagEngine.query()` or call `RetrievalService.retrieve(..., trace=True)` directly to receive a `RetrievalTrace` (in `retrieval/common/models.py`) capturing the full per-stage state: `query`, `rewritten_queries`, `per_method_results` (keyed by `BaseRetrievalMethod.name`, includes empty-result methods), `fused_results`, `reranked_results`, `refined_results`, `final_results`, `grounding_decision`, `confidence`, `routing_decision` (R1 placeholder), `timings`, `knowledge_id`. Default `trace=False` is byte-for-byte unchanged. The `None` vs `[]` distinction is load-bearing: `reranked_results is None` means "reranker not configured", `[]` means "ran with no input". `query_stream` does not collect a trace (deferred). Failure classification (R8.2) and benchmark harness (R8.3) build on top.

**Failure classification** (R8.2, 2026-04-27) — `classify_failure(query, trace) -> FailureClassification` in `retrieval/modules/evaluation/failure_analysis.py` (re-exported from `rfnry_rag.retrieval.modules.evaluation`). Pure inspection function on a `RetrievalTrace`: returns one of seven `FailureType` verdicts (`VOCABULARY_MISMATCH`, `CHUNK_BOUNDARY`, `SCOPE_MISS`, `ENTITY_NOT_INDEXED`, `LOW_RELEVANCE`, `INSUFFICIENT_CONTEXT`, `UNKNOWN`) plus a `signals` dict reporting the trace-derived values that drove the verdict. First-match-wins priority order documented in the module docstring; test #8 in `test_failure_classification.py` is the regression guard. Heuristic-only (no LLM, no new deps); LLM-backed classifier deferred. Three module-private thresholds (`_VOCABULARY_MISMATCH_THRESHOLD=0.4`, `_HIGH_RELEVANCE_THRESHOLD=0.7`, `_LOW_RELEVANCE_THRESHOLD=0.3`) live at module scope, not on a config dataclass — promote to a `FailureAnalysisConfig` if a real consumer needs to tune them. Caller's responsibility to invoke only on failed cases. R8.3's benchmark harness aggregates the verdicts into a failure-distribution report.

**Benchmark harness** (R8.3, 2026-04-27) — `RagEngine.benchmark(cases) -> BenchmarkReport` (Python API) and `rfnry-rag benchmark cases.json -k <knowledge_id> [-o report.json]` (CLI). Pure orchestration over R8.1 traces + R8.2 classifications + the existing `ExactMatch` / `F1Score` / `LLMJudgment` / `RetrievalRecall` / `RetrievalPrecision` metrics — no new metric implementations and no new LLM calls beyond the existing ones. Public types live in `retrieval/modules/evaluation/benchmark.py` (re-exported from `rfnry_rag.retrieval.modules.evaluation` and from `rfnry_rag.retrieval`): `BenchmarkCase`, `BenchmarkConfig`, `BenchmarkCaseResult`, `BenchmarkReport`, `run_benchmark`. `failure_distribution` is a `dict[str, int]` keyed on `FailureType.name` (e.g. `"VOCABULARY_MISMATCH"`) — `FailureClassification` is `frozen=True` but contains a `dict[str, ...]`, so Python does not generate `__hash__` for it; aggregating must key on the name (also keeps the JSON `--output` human-readable). `retrieval_recall` / `retrieval_precision` are `None` when at least one case omits `expected_source_ids` (N/A is distinct from 0.0). Failure rule: F1 < `failure_threshold` (default 0.5) OR `trace.grounding_decision == "ungrounded"`. CLI exit code is 0 on success regardless of failure rate (treating failure-rate as a CI signal is a follow-up).

### Modular Pipeline

Retrieval and ingestion are protocol-based plugin architectures. No mandatory vector DB or embeddings — at least one retrieval path (vector, document, or graph) must be configured.

- **`BaseRetrievalMethod`** / **`BaseIngestionMethod`** — Protocol interfaces in `modules/retrieval/base.py` and `modules/ingestion/base.py`. Any conforming class works.
- **Method classes** — `VectorRetrieval`, `DocumentRetrieval`, `GraphRetrieval` (retrieval); `VectorIngestion`, `DocumentIngestion`, `GraphIngestion`, `TreeIngestion` (ingestion). Each is self-contained with error isolation and timing logs.
- **`MethodNamespace[T]`** — Generic container exposing methods as attributes (`rag.retrieval.vector`) and supporting iteration (`for m in rag.retrieval`).
- **Dynamic assembly** — `RagEngine.initialize()` builds method lists from config, validates cross-config constraints via `_validate_config()`, assembles `RetrievalService` and `IngestionService` with method list dispatch.
- **`AnalyzedIngestionService`** — 3-phase LLM pipeline (analyze → synthesize → ingest) for vision-analyzed documents. Uses `graph_store` directly for pre-extracted entities, delegates document storage to method list. Phase A4: produces **3 vector kinds per page** — `description` (LLM prose), `raw_text` (PyMuPDF OCR, if non-empty), `table_row` (one vector per table row, column-header-prefixed); each payload tagged by `vector_role`. `full_text` for downstream document-store methods is now raw OCR text (fallback: LLM description + entities for L5X/XML). Phase B: page analyses stored in dedicated `rag_page_analyses` table (B1/B2); **file-hash + per-page-hash caching** short-circuits redundant LLM calls on re-ingestion (B3); **status-based resume** routes to the first unfinished phase on restart (B4); **PyMuPDF text-density pre-filter** skips vision LLM calls for text-dense pages (B5); concurrency configurable via `IngestionConfig.analyze_concurrency` (B6).
- **`DrawingIngestionService`** — sibling of `AnalyzedIngestionService` for
  drawing-first documents (schematics, P&ID, wiring, mechanical). 4-phase
  pipeline (`render → extract → link → ingest`) ingesting both PDF
  (vision-based via `AnalyzeDrawingPage`) and DXF (zero-LLM direct parse via
  ezdxf). The `link` phase resolves cross-sheet connectivity
  deterministically (exact off-page tags, regex target hints, RapidFuzz
  label merges) with LLM residue via `SynthesizeDrawingSet` only when
  unresolved candidates remain. Symbol vocabularies, off-page-connector
  regexes, and wire_style→relation_type mapping are fully consumer-
  configurable via `DrawingIngestionConfig` (ships IEC 60617 + ISA 5.1
  defaults). The DXF extractor scans modelspace `TEXT` + `MTEXT` against
  `config.off_page_connector_patterns` to populate
  `DrawingPageAnalysis.off_page_connectors`, anchoring each tag to the
  underlying component when its insertion point sits inside a component
  bbox (Phase F3.1, 2026-04-26). DXF render + extract iterate every
  layout — modelspace as `page_number=1`, then paperspace layouts in
  DXF tab order (`doc.layouts.names_in_taborder()`, Model alias
  skipped) — so multi-sheet drawings emit one page per layout instead
  of silently dropping per-sheet content (Phase F3.2, 2026-04-25).
  Phase C (2026-04-25).
- **Graph ingestion is consumer-agnostic by default.** The analyze-path graph
  mapper at `stores/graph/mapper.py` takes a `GraphIngestionConfig` so
  consumers supply their own entity-type regex patterns, relationship
  keyword map, and fallback edge type. Empty config → type-inference
  falls through to `DiscoveredEntity.category.lower()`, cross-references
  with no keyword match become generic `MENTIONS` edges. Phase D
  (2026-04-26).

### Error Hierarchy

```
SdkBaseError (common base — not re-exported at top level; catch subclasses)
├── ConfigurationError (shared)
├── RagError (retrieval)
│   ├── IngestionError, ParseError, EmptyDocumentError, EmbeddingError, IngestionInterruptedError, TreeIndexingError
│   ├── RetrievalError, GenerationError, TreeSearchError
│   └── StoreError, DuplicateSourceError, SourceNotFoundError
└── ReasoningError (reasoning)
    ├── AnalysisError, ClassificationError, ClusteringError
    ├── ComplianceError, EvaluationError
    └── ReasoningInputError(ReasoningError, ValueError) — input-validation failures in reasoning configs; back-compat with `except ValueError:` preserved
```

Note: `SdkBaseError` was previously named `BaseException`; it was renamed to avoid shadowing the Python builtin. Top-level `rfnry_rag` does NOT re-export it — catch the specific subclasses or import from `rfnry_rag.common.errors` directly.

### LLM Integration

All LLM calls go through BAML for structured output parsing, retry/fallback policies, and observability. Each SDK has its own `baml_src/` (source definitions) and `baml_client/` (auto-generated — never edit). After modifying `.baml` files, regenerate with `poe baml:generate:retrieval` or `poe baml:generate:reasoning`.

`LanguageModelClient` in `common/language_model.py` builds a BAML `ClientRegistry` with primary + optional fallback provider routing. `LanguageModelProvider` configures a single provider endpoint (API key, base URL, model).

## Key Patterns

- **Protocol-based abstraction** — No inheritance; `Protocol` classes define interfaces (`BaseEmbeddings` (in `rfnry_rag.common.protocols`), `BaseSemanticIndex`, `BaseReranking`, `BaseRetrievalMethod`, `BaseIngestionMethod`, `BaseQueryRewriting`, `BaseChunkRefinement`, `BaseRetrievalJudgment`, etc.). Any conforming object works.
- **Facade pattern** — `Embeddings(LanguageModelProvider)`, `Vision(LanguageModelProvider)`, and `Reranking(LanguageModelProvider | LanguageModelClient)` are public facades that select the correct private provider implementation at runtime. Concrete providers (e.g. OpenAI, Voyage, Cohere) are private. `Vision` dispatches to `_AnthropicVision`, `_OpenAIVision`, or `_GeminiVision` (Phase F3.5, 2026-04-25) based on `LanguageModelProvider.provider`.
- **Modular pipeline** — Retrieval and ingestion methods are pluggable. Services receive `list[BaseRetrievalMethod]` / `list[BaseIngestionMethod]` and dispatch generically. Methods carry `weight` and `top_k` configuration. Per-method error isolation (catch, log, continue).
- **Async-first** — All I/O is async. Services use `async def`, stores use asyncpg/aiosqlite.
- **Service pattern** — Each module has a `Service` class with dependencies injected via `__init__`.
- **Shared common, SDK-specific re-exports** — SDK `common/` modules are thin re-exports from `rfnry_rag.common`. Retrieval-specific utilities (models, formatting, hashing, page_range) stay in retrieval's own `common/`.
- **Config dataclasses** — Pydantic V2 or plain dataclasses with `__post_init__` validation. `PersistenceConfig.vector_store` and `IngestionConfig.embeddings` are optional — at least one retrieval path must be configured.

## Contract tests

The following contract tests act as regression guards — they enforce whole-class invariants so that future rounds don't rediscover the same issue categories:

- `test_baml_prompt_fence_contract.py` — every user-controlled BAML prompt parameter across retrieval + reasoning must be fenced. Fails if any new `.baml` file introduces an unfenced interpolation.
- `test_config_bounds_contract.py` (retrieval + reasoning) — every `int` / `float` field in a config dataclass must have a `__post_init__` bounds check or carry a `# unbounded: <reason>` marker. Fails if a new config field is added without explicit validation.
- `test_no_bare_valueerror_in_configs.py` — reasoning config `__post_init__` methods must raise `ReasoningInputError`, not bare `ValueError`. Enforces the error hierarchy so callers can distinguish validation failures from unexpected errors.

## Linting & Style

- Ruff: line-length 120, target py312, rules: E, F, I, UP, B, SIM, RUF022
- MyPy: python 3.12, ignores missing imports
- Both tools exclude `baml_client/` directories

## Testing

- pytest with `asyncio_mode = "auto"` — no `@pytest.mark.asyncio` needed
- Tests use `AsyncMock` and `SimpleNamespace` for lightweight mocking
- Tests in `tests/` subdirectories within each SDK + inline `test_*.py` in some modules
- 1060 tests total across both SDKs (Phase F adds +21 vs. the Phase E baseline of 997: F2 DrawingIngestionService status-transition + re-entry tests, F3.1 DXF TEXT/MTEXT off-page connectors, F3.2 DXF paperspace layout rendering, F3.5 Gemini vision provider; R4 adds +9 chunk-ordering unit cases; R3 adds +8 document-expansion unit cases; R8.1 adds +9 retrieval-trace unit cases; R8.2 adds +8 failure-classification unit cases; R8.3 adds +8 benchmark-harness unit + CLI cases (6 unit + 1 CLI + 1 LLM-judge unit))

## Config defaults and enforced bounds

`__post_init__` validators reject pathological values at construction time, not runtime:

- `IngestionConfig.dpi`: `72 ≤ dpi ≤ 600`
- `IngestionConfig.analyze_concurrency`: `1 ≤ n ≤ 100`, default 5 (Phase B6)
- `IngestionConfig.analyze_text_skip_threshold_chars`: `0 ≤ n ≤ 100_000`, default 300; 0 disables text-density pre-filter (Phase B5)
- `IngestionConfig.chunk_context_headers` (was `contextual_chunking`, old name deprecated)
- `IngestionConfig.chunk_size_unit`: `Literal["chars", "tokens"]`, default `"tokens"` (Phase A1). Default `chunk_size=375` tokens (was 500 chars), `chunk_overlap=40` (was 50).
- `IngestionConfig.parent_chunk_size`: sentinel `-1` (default) resolves to `3 * chunk_size` in `__post_init__`; explicit `0` disables parent-child indexing (Phase A5).
- `IngestionConfig.document_expansion`: nested `DocumentExpansionConfig` (R3, 2026-04-26). Defaults disabled. When `enabled=True`, `lm_client` is required; bounds `num_queries ∈ [1, 20]` (default 5) and `concurrency ∈ [1, 100]` (default 5). `include_in_embeddings` / `include_in_bm25` independently gate which downstream consumer receives the synthetic-query block; `synthetic_queries` is always stored on `ChunkedContent` for transparency. Explicit gating methods `text_for_embedding(*, include_synthetic=...)` / `text_for_bm25(*, include_synthetic=...)` complement the backward-compatible `embedding_text` property.
- `BenchmarkConfig.concurrency`: `1 ≤ n ≤ 20`, default 1 (serial); larger benchmarks opt into parallelism. Bounded via `run_concurrent` (R8.3, 2026-04-27).
- `BenchmarkConfig.failure_threshold`: `0.0 ≤ t ≤ 1.0`, default 0.5; F1 below this counts a case as failed for `failure_distribution` aggregation (R8.3, 2026-04-27).
- `DrawingIngestionConfig.dpi`: `150 ≤ dpi ≤ 600`, default 400 (higher than
  analyze's 300 — schematics need legible line weights)
- `DrawingIngestionConfig.analyze_concurrency`: `1 ≤ n ≤ 100`, default 5
- `DrawingIngestionConfig.fuzzy_label_threshold`: `0.0 ≤ t ≤ 1.0`, default
  0.92
- `DrawingIngestionConfig.graph_write_batch_size`: `1 ≤ n ≤ 10_000`,
  default 500
- `DrawingIngestionConfig.relation_vocabulary`: every target must be in
  `ALLOWED_RELATION_TYPES` (validated at `__post_init__`)
- `GraphIngestionConfig.entity_type_patterns`: list of
  `(regex_str, type_name)`; regex strings compiled at `__post_init__` for
  fail-fast validation
- `GraphIngestionConfig.relationship_keyword_map`: all values must be in
  `ALLOWED_RELATION_TYPES`
- `GraphIngestionConfig.unclassified_relation_default`: `"MENTIONS"` by
  default; `None` = drop; any other value must be in
  `ALLOWED_RELATION_TYPES`
- `RetrievalConfig.top_k`: `1 ≤ top_k ≤ 200`
- `RetrievalConfig.bm25_max_chunks`: `≤ 200_000`
- `RetrievalConfig.bm25_max_indexes`: `1 ≤ n ≤ 1000`, default 16
- `TreeSearchConfig.max_sources_per_query`: `1 ≤ n ≤ 1000`, default 50
- `TreeSearchConfig.max_steps`: `≤ 50`
- `TreeSearchConfig.max_context_tokens`: `≤ 500_000`
- `TreeIndexingConfig.toc_scan_pages`: `≤ 500`
- `TreeIndexingConfig.max_pages_per_node`: `≤ 200`
- `TreeIndexingConfig.max_tokens_per_node`: `≤ 200_000`
- `GenerationConfig`: `grounding_enabled=True` requires `grounding_threshold > 0` and an `lm_client`
- `GenerationConfig.chunk_ordering`: `ChunkOrdering` enum, default `SCORE_DESCENDING`; opt-in `PRIMACY_RECENCY` or `SANDWICH` for Lost-in-the-Middle mitigation (R4, 2026-04-25)
- Cross-config: `tree_indexing.max_tokens_per_node ≤ tree_search.max_context_tokens`
- `BatchConfig.batch_size`: `≤ 100_000`
- `BatchConfig.concurrency`: `1 ≤ concurrency ≤ 20`
- `run_concurrent` (common concurrency helper): `1 ≤ concurrency ≤ 100` (separate from `BatchConfig.concurrency` which is capped at 20)
- `RetrievalConfig.history_window`: `1 ≤ history_window ≤ 20`, default 3
- `QdrantVectorStore.hybrid_prefetch_multiplier`: `≥ 1`, default 4
- `PostgresDocumentStore.headline_max_words` / `headline_min_words` / `headline_max_fragments`: `≥ 1`, with `min_words ≤ max_words`
- `LanguageModelClient.timeout_seconds`: `> 0`, default `60`
- `Neo4jGraphStore.password`: required (no default; empty raises)
- `RetrievalConfig.source_type_weights`: each value in `(0, 10]`
- `MultiQueryRewriting.num_variants`: `1–10`
- `LanguageModelClient.temperature`: `0.0 ≤ temperature ≤ 2.0`
- `ClassificationConfig.concurrency`: `1–20`
- `ClusteringConfig.n_clusters`: `2–1000` (K-Means)
- `ClusteringConfig.min_cluster_size`: `2–10_000` (HDBSCAN)
- `ComplianceConfig.concurrency`: `1–20`
- `ComplianceConfig.max_reference_length`: `1–5_000_000`
- `TreeIndex.pages`: `≤ 100_000` (security cap in `from_dict`)
- `_MAX_PAGES_PER_ENTITY = 20` in analyzed ingestion (cross-ref pairwise expansion cap)
- Public-input bounds: query ≤ 32 000 chars, `ingest_text` ≤ 5 000 000 chars, metadata ≤ 50 keys × 8 000 chars

Per-op timeouts (all configurable):

- `LanguageModelClient.timeout_seconds`: 60
- `QdrantVectorStore.timeout` / `scroll_timeout` / `write_timeout`: 10 / 30 / 30
- `Neo4jGraphStore.connection_timeout` / `connection_acquisition_timeout`: 5.0 / 5.0
- `SQLAlchemyMetadataStore.pool_timeout` / `PostgresDocumentStore.pool_timeout`: 10
- `BOUNDARY_API_KEY` collisions across `LanguageModelClient` instances raise `ConfigurationError` (first-write-wins)

## Ingestion method contract

`BaseIngestionMethod.required: bool` is part of the protocol (not optional). `VectorIngestion` and `DocumentIngestion` default `required=True`; `GraphIngestion` and `TreeIngestion` default `required=False`. Required-method failures abort the ingest with `IngestionError` and skip the metadata commit (no partial-success row written).

## Environment Variables

- `RFNRY_RAG_LOG_ENABLED=true` / `RFNRY_RAG_LOG_LEVEL=DEBUG` — SDK logging
- `RFNRY_RAG_LOG_QUERIES=true` — include raw query text in logs (off by default; PII-safe). Use `rfnry_rag.common.logging.query_logging_enabled()` when adding new query-logging sites.
- `RFNRY_RAG_BAML_LOG=info|warn|debug` — BAML runtime logging (SDK sets `BAML_LOG` from this)
- `BAML_LOG=info|warn|debug` — BAML runtime logging (direct override)
- `BOUNDARY_API_KEY` — Boundary collector key, process-global
- Config lives at `~/.config/rfnry_rag/config.toml` + `.env`
