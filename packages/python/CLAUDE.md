# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

rfnry-rag is a dual-SDK Python package providing two AI pipelines:
- **Retrieval SDK** (`src/rfnry_rag/retrieval/`) — Document ingestion, multi-path semantic search, LLM-grounded generation.
- **Reasoning SDK** (`src/rfnry_rag/reasoning/`) — Text analysis, classification, clustering, compliance checking, evaluation, pipeline composition.

Both SDKs share common infrastructure in `src/rfnry_rag/common/` (errors, language model config, logging, concurrency, CLI utilities). Each SDK has its own `common/` that re-exports from the shared common — never duplicate code between them.

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

Run a single test: `pytest tests/retrieval/test_search.py::test_name -v`

## Architecture

### Package Structure

```
src/rfnry_rag/
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
│   ├── common/          # Re-exports + retrieval-specific (models, formatting, hashing, page_range, grounding)
│   ├── server.py        # RagEngine — main entry point, dynamic pipeline assembly
│   ├── modules/
│   │   ├── namespace.py # MethodNamespace[T] — attribute access + iteration for pipeline methods
│   │   ├── ingestion/   # base.py protocol, methods/ (vector, document, graph, tree, raptor), chunk/, analyze/, embeddings/, vision/, tree/
│   │   ├── retrieval/   # base.py protocol, methods/ (vector, document, graph, raptor), search/ (service, fusion, classification, reranking, rewriting), refinement/, enrich/, judging, tree/, iterative/
│   │   ├── generation/  # service, step, grounding, confidence
│   │   ├── knowledge/   # manager (CRUD), migration
│   │   └── evaluation/  # metrics (ExactMatch, F1, LLMJudgment), retrieval_metrics, failure_analysis, benchmark
│   ├── stores/          # vector/ (Qdrant), metadata/ (SQLAlchemy), document/ (Postgres, filesystem), graph/ (Neo4j)
│   ├── cli/             # Click commands, config loader, output formatters
│   └── baml/            # baml_src/ (edit) + baml_client/ (generated, do not edit)
└── reasoning/
    ├── common/          # Re-exports from rfnry_rag.common
    ├── modules/
    │   ├── analysis/      # AnalysisService — intent, dimensions, entities, context tracking
    │   ├── classification/# ClassificationService — LLM or hybrid kNN→LLM
    │   ├── clustering/    # ClusteringService — K-Means, HDBSCAN, LLM labeling
    │   ├── compliance/    # ComplianceService — policy violation checking
    │   ├── evaluation/    # EvaluationService — similarity + LLM judge scoring
    │   └── pipeline/      # Pipeline — sequential step composition
    ├── protocols.py     # BaseEmbeddings (from rfnry_rag.common.protocols), BaseSemanticIndex (structural typing)
    ├── cli/
    └── baml/
```

### Metadata store schema

`SQLAlchemyMetadataStore` owns four tables:

- `rag_schema_meta` — schema-version guard for forward migrations.
- `rag_sources` — per-source records keyed by source_id, indexed by file_hash.
- `rag_page_analyses` — per-page analysis cache, composite key (source_id, page_number), indexed by page_hash, FK cascade-delete.
- `rag_raptor_trees` — per-knowledge_id active-tree pointer for hierarchical summarization retrieval.

### Entry Points

- **Retrieval:** `RagEngine` in `server.py` — async context manager. `async with RagEngine(config) as rag:`
- **Reasoning:** Services are standalone (`AnalysisService`, `ClassificationService`, etc.). `Pipeline` composes them sequentially.
- **CLI:** `rfnry-rag retrieval <cmd>` / `rfnry-rag reasoning <cmd>` (also standalone: `rfnry-rag-retrieval`, `rfnry-rag-reasoning`).
- **SDK import:** `from rfnry_rag import RagEngine, Pipeline, AnalysisService` — top-level re-exports everything from both SDKs.

### Retrieval pipeline

`RagEngine.query()` runs:

1. **Mode dispatch.** `RoutingConfig.mode` selects between RETRIEVAL (the standard pipeline below), DIRECT (load full corpus into the prompt, skip chunk-level grounding/clarification), HYBRID (RAG-first, LLM-judged answerability check, escalate to direct-context on `answerable=False`), and AUTO (per-query token-count threshold picks retrieval-vs-direct via `KnowledgeManager.get_corpus_tokens`). Mode default is RETRIEVAL.

2. **Iterative gate** (when `IterativeRetrievalConfig.enabled`). Inside the RETRIEVAL arm, a query classifier optionally routes complex or entity-relationship queries through `IterativeRetrievalService`, which decomposes the query into sequential sub-questions and accumulates findings across hops. Gate fall-through proceeds with single-pass retrieval. Hops can post-loop-escalate to direct-context mode when accumulated chunks remain weak.

3. **Adaptive parameters** (when `AdaptiveRetrievalConfig.enabled`). At the top of `RetrievalService.retrieve` (before query rewriting, so the classifier sees the original query, not LLM-generated variants), `_compute_adaptive_params` reads the classifier verdict and produces an effective `top_k` plus per-method-weight multipliers keyed by query type.

4. **Query rewriting** (optional). HyDE, multi-query, or step-back. Expands one query into multiple variants via an LLM call. Configured via `RetrievalConfig.query_rewriter`.

5. **Multi-path search.** Configured retrieval methods run concurrently; results merge via reciprocal rank fusion with per-method weights:
   - `VectorRetrieval` — Dense similarity + SPLADE hybrid (when `sparse_embeddings`) + BM25 (when `bm25_enabled`), fused internally via RRF.
   - `DocumentRetrieval` — Full-text + substring search (requires document store).
   - `GraphRetrieval` — Entity lookup + N-hop traversal (requires graph store). Exposes `trace(entity_name, max_hops, relation_types, knowledge_id)` for programmatic queries returning `GraphPath` objects.
   - `RaptorRetrieval` — Searches summary vectors filtered to the active hierarchical-summarization tree for a knowledge_id (when `IngestionConfig.raptor.enabled`).
   - **Tree search** — When `TreeSearchConfig.enabled`, `_run_tree_search` loads stored tree indexes for up to `max_sources_per_query` sources, runs LLM-backed navigation per source concurrently, and injects results into the RRF fusion pool.

6. **Confidence expansion** (when `AdaptiveRetrievalConfig.confidence_expansion`). After the first retrieval attempt, if `max(score) < grounding_threshold`, retry with expanded `top_k` (capped at `top_k_max`). After `max_expansion_retries` exhausted with chunks still weak, optionally escalate to direct-context mode when corpus fits `direct_context_threshold`.

7. **Reranking** (optional). Cross-encoder reranking against the original query (Cohere, Voyage).

8. **Chunk refinement** (optional). Extractive (context window) or abstractive (LLM summarization).

9. **Generation.** Grounding gate → LLM relevance gate → optional clarification → LLM generation. Context assembly via `chunks_to_context()` accepts `GenerationConfig.chunk_ordering` (`SCORE_DESCENDING` default, `PRIMACY_RECENCY`, or `SANDWICH`) to mitigate the lost-in-the-middle effect.

### Optional trace

Pass `trace=True` to `RagEngine.query()` to receive a `RetrievalTrace` (in `retrieval/common/models.py`) capturing per-stage state: `query`, `rewritten_queries`, `per_method_results` (keyed by `BaseRetrievalMethod.name`, includes empty-result methods), `fused_results`, `reranked_results`, `refined_results`, `final_results`, `grounding_decision`, `confidence`, `routing_decision`, `adaptive`, `iterative_hops`, `iterative_termination_reason`, `timings`, `knowledge_id`. Default `trace=False` is byte-for-byte unchanged. The `None` vs `[]` distinction is load-bearing: `reranked_results is None` means "reranker not configured", `[]` means "ran with no input". `query_stream` does not collect a trace.

### Failure classification + benchmark harness

`classify_failure(query, trace) -> FailureClassification` (in `modules/evaluation/failure_analysis.py`) is a pure heuristic inspection on a `RetrievalTrace`. Returns one of seven `FailureType` verdicts (`VOCABULARY_MISMATCH`, `CHUNK_BOUNDARY`, `SCOPE_MISS`, `ENTITY_NOT_INDEXED`, `LOW_RELEVANCE`, `INSUFFICIENT_CONTEXT`, `UNKNOWN`) plus a `signals` dict reporting the trace-derived values that drove the verdict. First-match-wins priority order documented in the module docstring. Caller invokes only on failed cases.

`RagEngine.benchmark(cases) -> BenchmarkReport` (and CLI `rfnry-rag benchmark cases.json -k <knowledge_id>`) orchestrates traces + classifications + the existing `ExactMatch` / `F1Score` / `LLMJudgment` / `RetrievalRecall` / `RetrievalPrecision` metrics. `failure_distribution` keyed on `FailureType.name`. `retrieval_recall` / `retrieval_precision` are `None` when at least one case omits `expected_source_ids` (N/A is distinct from 0.0). Failure rule: F1 < `failure_threshold` (default 0.5) OR `trace.grounding_decision == "ungrounded"`.

### Hierarchical summarization (RAPTOR)

When `IngestionConfig.raptor.enabled`, the consumer can call `RagEngine.build_raptor_index(knowledge_id) -> RaptorBuildReport` to build a per-knowledge_id summary tree. The builder loads leaf vectors filtered to chunk-based vector roles (drawing components excluded), clusters them (K-Means default, HDBSCAN opt-in), summarizes each cluster concurrently (capped at the centroid-nearest 20 members per call), embeds the summaries, persists them with `vector_role="raptor_summary"` plus `raptor_tree_id`, `raptor_level`, `raptor_parent_id`, `raptor_cluster_size`, and recurses up to `max_levels`. Atomic blue/green swap: persist all summaries → `set_active` on `RaptorTreeRegistry` → garbage-collect old summary vectors. `RaptorRetrieval` (registered into `MethodNamespace` when raptor is enabled) reads the active tree pointer and searches summary nodes filtered to the active tree.

### Modular pipeline

Retrieval and ingestion are protocol-based plugin architectures. No mandatory vector DB or embeddings — at least one retrieval path (vector, document, or graph) must be configured.

- **`BaseRetrievalMethod` / `BaseIngestionMethod`** — Protocol interfaces in `modules/retrieval/base.py` and `modules/ingestion/base.py`.
- **Method classes** — `VectorRetrieval`, `DocumentRetrieval`, `GraphRetrieval`, `RaptorRetrieval` (retrieval); `VectorIngestion`, `DocumentIngestion`, `GraphIngestion`, `TreeIngestion` (ingestion). Each is self-contained with error isolation and timing logs.
- **`MethodNamespace[T]`** — Generic container exposing methods as attributes (`rag.retrieval.vector`) and supporting iteration (`for m in rag.retrieval`).
- **Dynamic assembly** — `RagEngine.initialize()` builds method lists from config, validates cross-config constraints via `_validate_config()`, assembles `RetrievalService` and `IngestionService` with method list dispatch.
- **`AnalyzedIngestionService`** — multi-phase LLM pipeline (analyze → synthesize → ingest) for vision-analyzed documents. Produces three vector kinds per page (`description` LLM prose, `raw_text` PyMuPDF OCR if non-empty, `table_row` per row column-header-prefixed); each payload tagged by `vector_role`. Page analyses cached by file-hash + per-page-hash with status-based resume; PyMuPDF text-density pre-filter skips vision LLM calls for text-dense pages; concurrency configurable via `IngestionConfig.analyze_concurrency`.
- **`DrawingIngestionService`** — sibling of `AnalyzedIngestionService` for drawing-first documents (schematics, P&ID, wiring, mechanical). Four-phase pipeline (`render → extract → link → ingest`) ingesting both PDF (vision-based via `AnalyzeDrawingPage`) and DXF (zero-LLM direct parse via `ezdxf`, modelspace + paperspace layouts in tab order). The `link` phase resolves cross-sheet connectivity deterministically (off-page tags, regex hints, RapidFuzz label merges) with LLM residue via `SynthesizeDrawingSet` only when unresolved candidates remain. Symbol vocabularies, off-page-connector regexes, and `wire_style → relation_type` mapping are consumer-configurable via `DrawingIngestionConfig` (ships IEC 60617 + ISA 5.1 defaults).
- **Graph ingestion is consumer-agnostic by default.** The analyze-path graph mapper at `stores/graph/mapper.py` takes a `GraphIngestionConfig` so consumers supply their own entity-type regex patterns, relationship keyword map, and fallback edge type. Empty config → type inference falls through to `DiscoveredEntity.category.lower()`; cross-references with no keyword match become generic `MENTIONS` edges.

### Error hierarchy

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

`SdkBaseError` is intentionally not re-exported at the top level — catch the specific subclasses or import from `rfnry_rag.common.errors` directly. The name avoids shadowing Python's builtin `BaseException`.

### LLM integration

All LLM calls go through BAML for structured output parsing, retry/fallback policies, and observability. Each SDK has its own `baml_src/` (source definitions) and `baml_client/` (auto-generated — never edit). After modifying `.baml` files, regenerate with `poe baml:generate:retrieval` or `poe baml:generate:reasoning`.

`LanguageModelClient` in `common/language_model.py` builds a BAML `ClientRegistry` with primary + optional fallback provider routing. `LanguageModelProvider` configures a single provider endpoint (API key, base URL, model).

## Key patterns

- **Protocol-based abstraction.** No inheritance; `Protocol` classes define interfaces (`BaseEmbeddings` (in `rfnry_rag.common.protocols`), `BaseSemanticIndex`, `BaseReranking`, `BaseRetrievalMethod`, `BaseIngestionMethod`, `BaseQueryRewriting`, `BaseChunkRefinement`, `BaseRetrievalJudgment`). Any conforming object works.
- **Facade pattern.** `Embeddings(LanguageModelProvider)`, `Vision(LanguageModelProvider)`, and `Reranking(LanguageModelProvider | LanguageModelClient)` are public facades that select the correct private provider implementation at runtime. `Vision` dispatches to `_AnthropicVision`, `_OpenAIVision`, or `_GeminiVision` based on `LanguageModelProvider.provider`.
- **Modular pipeline.** Retrieval and ingestion methods are pluggable. Services receive `list[BaseRetrievalMethod]` / `list[BaseIngestionMethod]` and dispatch generically. Methods carry `weight` and `top_k` configuration. Per-method error isolation (catch, log, continue).
- **Async-first.** All I/O is async. Services use `async def`, stores use asyncpg / aiosqlite.
- **Service pattern.** Each module has a `Service` class with dependencies injected via `__init__`.
- **Shared common, SDK-specific re-exports.** SDK `common/` modules are thin re-exports from `rfnry_rag.common`. Retrieval-specific utilities (models, formatting, hashing, page_range, grounding) stay in retrieval's own `common/`.
- **Config dataclasses.** Pydantic V2 or plain dataclasses with `__post_init__` validation. `PersistenceConfig.vector_store` and `IngestionConfig.embeddings` are optional — at least one retrieval path must be configured.
- **Domain-neutral by default.** No hardcoded domain vocabulary in BAML prompts. Features needing vocabulary (entity types, relationship keywords, symbol libraries, relation-type maps) expose consumer-overridable config with empty defaults; values are validated against an allowlist where applicable.

## Contract tests

The following contract tests act as regression guards — they enforce whole-class invariants:

- `test_baml_prompt_fence_contract.py` — every user-controlled BAML prompt parameter across retrieval + reasoning must be fenced. Fails if any new `.baml` file introduces an unfenced interpolation. Supports both single-tag interpolation and Jinja-loop multi-tag interpolation (e.g. `{% for member_text in cluster_texts %} ... ======== MEMBER_{{ loop.index }} START ========`).
- `test_baml_prompt_domain_agnostic.py` — scans `baml_src/retrieval/functions.baml` for domain-bias vocabulary; fails on any banned term.
- `test_config_bounds_contract.py` (retrieval + reasoning) — every `int` / `float` field in a config dataclass must have a `__post_init__` bounds check or carry a `# unbounded: <reason>` marker.
- `test_no_bare_valueerror_in_configs.py` — reasoning config `__post_init__` methods must raise `ReasoningInputError`, not bare `ValueError`. Retrieval configs raise `ConfigurationError`.

## Linting & style

- Ruff: line-length 120, target py312, rules: E, F, I, UP, B, SIM, RUF022.
- MyPy: python 3.12, ignores missing imports.
- Both tools exclude `baml_client/` directories.

## Testing

- pytest with `asyncio_mode = "auto"` — no `@pytest.mark.asyncio` needed.
- Tests use `AsyncMock` and `SimpleNamespace` for lightweight mocking.
- Tests live in `tests/` at the package root, mirroring the dual-SDK split: `tests/common/`, `tests/reasoning/`, `tests/retrieval/`.

## Config defaults and enforced bounds

`__post_init__` validators reject pathological values at construction time, not runtime:

- `IngestionConfig.dpi`: `72 ≤ dpi ≤ 600`.
- `IngestionConfig.analyze_concurrency`: `1 ≤ n ≤ 100`, default 5.
- `IngestionConfig.analyze_text_skip_threshold_chars`: `0 ≤ n ≤ 100_000`, default 300; 0 disables text-density pre-filter.
- `IngestionConfig.chunk_size_unit`: `Literal["chars", "tokens"]`, default `"tokens"`. Default `chunk_size=375` tokens, `chunk_overlap=40`.
- `IngestionConfig.parent_chunk_size`: sentinel `-1` (default) resolves to `3 * chunk_size` in `__post_init__`; explicit `0` disables parent-child indexing.
- `IngestionConfig.chunk_context_headers` (legacy alias `contextual_chunking` deprecated).
- `IngestionConfig.document_expansion`: nested `DocumentExpansionConfig`. Defaults disabled. When `enabled=True`, `lm_client` is required; bounds `num_queries ∈ [1, 20]` (default 5) and `concurrency ∈ [1, 100]` (default 5). `include_in_embeddings` / `include_in_bm25` independently gate which downstream consumer receives the synthetic-query block; `synthetic_queries` is always stored on `ChunkedContent` for transparency.
- `BenchmarkConfig.concurrency`: `1 ≤ n ≤ 20`, default 1 (serial).
- `BenchmarkConfig.failure_threshold`: `0.0 ≤ t ≤ 1.0`, default 0.5.
- `DrawingIngestionConfig.dpi`: `150 ≤ dpi ≤ 600`, default 400.
- `DrawingIngestionConfig.analyze_concurrency`: `1 ≤ n ≤ 100`, default 5.
- `DrawingIngestionConfig.fuzzy_label_threshold`: `0.0 ≤ t ≤ 1.0`, default 0.92.
- `DrawingIngestionConfig.graph_write_batch_size`: `1 ≤ n ≤ 10_000`, default 500.
- `DrawingIngestionConfig.relation_vocabulary`: every target must be in `ALLOWED_RELATION_TYPES` (validated at `__post_init__`).
- `GraphIngestionConfig.entity_type_patterns`: list of `(regex_str, type_name)`; regex strings compiled at `__post_init__` for fail-fast validation.
- `GraphIngestionConfig.relationship_keyword_map`: all values must be in `ALLOWED_RELATION_TYPES`.
- `GraphIngestionConfig.unclassified_relation_default`: `"MENTIONS"` by default; `None` = drop; any other value must be in `ALLOWED_RELATION_TYPES`.
- `RetrievalConfig.top_k`: `1 ≤ top_k ≤ 200`.
- `RetrievalConfig.bm25_max_chunks`: `≤ 200_000`.
- `RetrievalConfig.bm25_max_indexes`: `1 ≤ n ≤ 1000`, default 16.
- `RetrievalConfig.history_window`: `1 ≤ n ≤ 20`, default 3.
- `RetrievalConfig.source_type_weights`: each value in `(0, 10]`.
- `TreeSearchConfig.max_sources_per_query`: `1 ≤ n ≤ 1000`, default 50.
- `TreeSearchConfig.max_steps`: `≤ 50`.
- `TreeSearchConfig.max_context_tokens`: `≤ 500_000`.
- `TreeIndexingConfig.toc_scan_pages`: `≤ 500`.
- `TreeIndexingConfig.max_pages_per_node`: `≤ 200`.
- `TreeIndexingConfig.max_tokens_per_node`: `≤ 200_000`.
- `GenerationConfig`: `grounding_enabled=True` requires `grounding_threshold > 0` and an `lm_client`.
- `GenerationConfig.chunk_ordering`: `ChunkOrdering` enum, default `SCORE_DESCENDING`; `PRIMACY_RECENCY` and `SANDWICH` are opt-in for lost-in-the-middle mitigation.
- `RoutingConfig.mode`: `QueryMode` enum, default `RETRIEVAL`. Other values: `DIRECT`, `HYBRID` (requires `hybrid_answerability_model`), `AUTO`.
- `RoutingConfig.direct_context_threshold`: `1_000 ≤ n ≤ 2_000_000`, default 150_000 (Anthropic's tested boundary for prompt-cached long-context performance; AUTO routes `tokens <= threshold` to DIRECT).
- `AdaptiveRetrievalConfig`: `enabled` defaults False (plumbing only). `top_k_min` `1 ≤ n ≤ 50`, default 3. `top_k_max` `1 ≤ n ≤ 200`, default 15. `top_k_min ≤ top_k_max` enforced. `max_expansion_retries` `0 ≤ n ≤ 5`, default 2. `confidence_expansion` defaults False. `use_llm_classification` cross-config requires `RetrievalConfig.enrich_lm_client` (validated in `RagEngine._validate_config`). `task_weight_profiles` is unbounded — consumer-supplied per-method overrides normalised by fusion at consume time.
- `IterativeRetrievalConfig`: `enabled` defaults False (byte-for-byte unchanged when off). `max_hops` `1 ≤ n ≤ 10`, default 3. `gate_mode` allowlist `{"type", "llm"}`, default `"type"`. `escalate_to_direct` defaults True. `grounding_threshold` `0.0 ≤ t ≤ 1.0` when set, default `None` (inherits `GenerationConfig.grounding_threshold` at consume time — no hardcoded fallback). Cross-field rule: `enabled=True AND gate_mode="llm"` requires `decomposition_model`. Cross-config rule: `enabled=True` requires either `decomposition_model` or `RetrievalConfig.enrich_lm_client`. Init-time warning: `escalate_to_direct=True` with `RoutingConfig.direct_context_threshold` unconfigured logs once at init (escalation is a no-op but consumers may intentionally run iterative without DIRECT fallback). Lives at `modules/retrieval/iterative/config.py`. Raises `ConfigurationError` on validation failures.
- `RaptorConfig`: `enabled` defaults False. `max_levels` `1 ≤ n ≤ 10`, default 3. `cluster_algorithm` allowlist `{"kmeans", "hdbscan"}`, default `"kmeans"`. `clusters_per_level` `2 ≤ n ≤ 100`, default 10. `min_cluster_size` `2 ≤ n ≤ 100`, default 5. `summary_max_tokens` `50 ≤ n ≤ 2000`, default 256. `summary_model: LanguageModelClient | None = None`. Cross-field rule: `enabled=True` requires `summary_model`. Lives at `modules/ingestion/methods/raptor/config.py`. Raises `ConfigurationError`.
- Cross-config: `tree_indexing.max_tokens_per_node ≤ tree_search.max_context_tokens`.
- `BatchConfig.batch_size`: `≤ 100_000`.
- `BatchConfig.concurrency`: `1 ≤ n ≤ 20`.
- `run_concurrent` (common concurrency helper): `1 ≤ concurrency ≤ 100` (separate from `BatchConfig.concurrency`'s 20-cap).
- `QdrantVectorStore.hybrid_prefetch_multiplier`: `≥ 1`, default 4.
- `PostgresDocumentStore.headline_max_words` / `headline_min_words` / `headline_max_fragments`: `≥ 1`, with `min_words ≤ max_words`.
- `LanguageModelClient.timeout_seconds`: `> 0`, default 60.
- `LanguageModelClient.temperature`: `0.0 ≤ t ≤ 2.0`.
- `Neo4jGraphStore.password`: required (no default; empty raises).
- `MultiQueryRewriting.num_variants`: `1–10`.
- `ClassificationConfig.concurrency`: `1–20`.
- `ClusteringConfig.n_clusters`: `2–1000` (K-Means).
- `ClusteringConfig.min_cluster_size`: `2–10_000` (HDBSCAN).
- `ComplianceConfig.concurrency`: `1–20`.
- `ComplianceConfig.max_reference_length`: `1–5_000_000`.
- `TreeIndex.pages`: `≤ 100_000` (security cap in `from_dict`).
- `_MAX_PAGES_PER_ENTITY = 20` in analyzed ingestion (cross-ref pairwise expansion cap).
- Public-input bounds: query ≤ 32 000 chars, `ingest_text` ≤ 5 000 000 chars, metadata ≤ 50 keys × 8 000 chars.

Per-op timeouts (all configurable):

- `LanguageModelClient.timeout_seconds`: 60.
- `QdrantVectorStore.timeout` / `scroll_timeout` / `write_timeout`: 10 / 30 / 30.
- `Neo4jGraphStore.connection_timeout` / `connection_acquisition_timeout`: 5.0 / 5.0.
- `SQLAlchemyMetadataStore.pool_timeout` / `PostgresDocumentStore.pool_timeout`: 10.
- `BOUNDARY_API_KEY` collisions across `LanguageModelClient` instances raise `ConfigurationError` (first-write-wins).

## Ingestion method contract

`BaseIngestionMethod.required: bool` is part of the protocol (not optional). `VectorIngestion` and `DocumentIngestion` default `required=True`; `GraphIngestion` and `TreeIngestion` default `required=False`. Required-method failures abort the ingest with `IngestionError` and skip the metadata commit (no partial-success row written).

## Environment variables

- `RFNRY_RAG_LOG_ENABLED=true` / `RFNRY_RAG_LOG_LEVEL=DEBUG` — SDK logging.
- `RFNRY_RAG_LOG_QUERIES=true` — include raw query text in logs (off by default; PII-safe). Use `rfnry_rag.common.logging.query_logging_enabled()` when adding new query-logging sites.
- `RFNRY_RAG_BAML_LOG=info|warn|debug` — BAML runtime logging (SDK sets `BAML_LOG` from this).
- `BAML_LOG=info|warn|debug` — BAML runtime logging (direct override).
- `BOUNDARY_API_KEY` — Boundary collector key, process-global.
- Config lives at `~/.config/rfnry_rag/config.toml` + `.env`.
