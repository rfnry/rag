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
                          # SQLAlchemyMetadataStore schema: rag_schema_meta (version/migration guard), rag_sources (source records + file_hash index), rag_page_analyses (per-page cache: composite key source_id+page_number, indexed page_hash, FK cascade-delete; Phase B1), rag_raptor_trees (per-knowledge_id active-tree pointer with built_at/level_counts/total_cost_usd; primary key knowledge_id; R2.1)
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

**Token counting + corpus loader** (R1.1, 2026-04-28) — invisible plumbing for the Phase 2 R1 routing series (R1.2/R1.3/R1.4 build on top). `Source.estimated_tokens: int | None` is a property reading `metadata["estimated_tokens"]` with `int(...)` coercion — stored in the metadata blob rather than a dedicated column so R1.1 ships without a schema migration. `IngestionService.ingest` and `ingest_text` populate the count at ingest time (sum of per-page `count_tokens` for files; single `count_tokens(content)` for text). `KnowledgeManager.get_corpus_tokens(knowledge_id) -> int` sums across sources; legacy rows lacking the count are lazy-computed from the document store and written back via `update_source(metadata=...)`. `RagEngine._load_full_corpus(knowledge_id) -> str` returns every source's text under `[Source: <name>]` separators — document store preferred, vector-store scroll as lossy fallback (skips `chunk_type == "parent"` to avoid double-counting Phase A5 parent-child indexing). Adds `BaseDocumentStore.get(source_id) -> str | None` to the protocol (Postgres + filesystem implementations). `mode="retrieval"` (the default) does not exercise any of this code; user-facing routing modes ship in R1.2/R1.3/R1.4.

**DIRECT context mode** (R1.2, 2026-04-28) — lights up `mode="direct"` user-facing. New `QueryMode` enum (`RETRIEVAL` / `DIRECT` / `HYBRID` / `AUTO` — string values match `RetrievalTrace.routing_decision` enumeration) and `RoutingConfig` dataclass at `rfnry_rag.retrieval.server` (re-exported from `rfnry_rag.retrieval`). `RagServerConfig.routing` defaults to `RoutingConfig()` with `mode=QueryMode.RETRIEVAL` so the existing pipeline is byte-for-byte unchanged for consumers who don't opt in. `RagEngine.query()` dispatches: RETRIEVAL → existing retrieve-then-generate (extracted to `_query_via_retrieval`); DIRECT → `_query_via_direct_context` which loads the full corpus via R1.1's `_load_full_corpus` and routes through new `GenerationService.generate_from_corpus` (skips both grounding and clarification gates by design — with the entire corpus in the prompt, the chunk-level relevance signal those gates depend on no longer applies, so running them would burn LLM calls without changing the outcome). HYBRID and AUTO raise `ConfigurationError` ("not yet implemented in R1.2; HYBRID lands in R1.3 and AUTO in R1.4") rather than silently falling back to RETRIEVAL — silent fallback would mask misconfiguration. `RetrievalTrace.routing_decision` (R8.1 placeholder) is now populated for the first time: `"retrieval"` or `"direct"`. `RoutingConfig.direct_context_threshold: int` bounded `1_000 ≤ n ≤ 2_000_000` (default 150_000) — R1.4's AUTO mode will read this; R1.2 declares it for forward compat. `RoutingConfig.hybrid_answerability_model: LanguageModelClient | None` declared with `None` default; R1.3 will enforce required-when-HYBRID. Trade-off: when DIRECT is enabled against a corpus exceeding model context, the LLM provider raises and the engine surfaces the error; R1.4's AUTO mode will pre-check `direct_context_threshold`. DIRECT returns `QueryResult(chunks=[])` (honest — no chunk-level source attribution from the full-corpus path).

**HYBRID mode (SELF-ROUTE)** (R1.3, 2026-04-28) — lights up `mode="hybrid"` user-facing. Implements the SELF-ROUTE strategy from Google DeepMind: run RAG first, ask the LLM whether the retrieved chunks are sufficient via new BAML function `CheckAnswerability(query, context) -> AnswerabilityVerdict { answerable: bool, reasoning: string }`, and only escalate to the full-corpus path when the verdict says no. `RoutingConfig.__post_init__` now enforces `hybrid_answerability_model` is non-`None` when `mode == HYBRID` (R1.2 declared the field for forward compat). `RagEngine.query()` dispatches `HYBRID` to `_query_via_hybrid`: (1) `_retrieve_chunks` (existing pipeline), (2) format via `chunks_to_context`, (3) `b.CheckAnswerability` against a registry built once in `_initialize_impl` from `RoutingConfig.hybrid_answerability_model`, (4) on `answerable=True` generate from chunks (`routing_decision="hybrid_rag"`), (5) on `answerable=False` load full corpus via R1.1's `_load_full_corpus` and generate via R1.2's `GenerationService.generate_from_corpus` (`routing_decision="hybrid_lc"`). `_check_answerability` is wrapped in try/except that logs a warning and returns `(True, "check_failed: <exc>")` on any failure — degrading to the cheaper RAG path rather than silently escalating to LC on a transient error (rate-limit, timeout, malformed JSON). `RetrievalTrace.routing_decision` now populates the remaining R8.1 enum values (`"hybrid_rag"` / `"hybrid_lc"`); combined with R1.2's `"retrieval"` / `"direct"`, the enum is fully realized. The new `CheckAnswerability` BAML function fences both `query` and `context` per Convention 3 (registered in `USER_CONTROLLED_PARAMS`) and is consumer-agnostic per Convention 1 (passes the F1 domain-bias contract). Per-query cost: HYBRID adds one cheap answerability LLM call (~$0.005, ~200 ms on Sonnet-class) on top of RAG. Verdict caching, streaming HYBRID, and a confidence-threshold knob are deferred. The `BAML.CheckAnswerability` import lives at `rfnry_rag.retrieval.server` so tests can `patch("rfnry_rag.retrieval.server.b.CheckAnswerability")`.

**Query classifier** (R5.1, 2026-04-28) — invisible plumbing for the Phase 2 R5 adaptive-retrieval series (R5.2 dynamic top_k + task-aware weights, R5.3 confidence expansion, R6 multi-hop iterative retrieval all build on top). Pure async `classify_query(text, lm_client=None) -> QueryClassification` in `modules/retrieval/search/classification.py` (re-exported from `rfnry_rag.retrieval`). Heuristic path (default, free, deterministic regex; ~µs) classifies on `query_type` (priority order ENTITY_RELATIONSHIP > COMPARATIVE > PROCEDURAL > FACTUAL) and `complexity` (length + entity-count + type-driven SIMPLE/MODERATE/COMPLEX). LLM path (opt-in via `lm_client`) routes through new BAML function `ClassifyQueryComplexity(query) -> QueryClassification { complexity, query_type, reasoning }` for higher accuracy on ambiguous text (~$0.001, ~200 ms). On LLM exception, logs a warning and falls back to the heuristic — `classify_query` never raises so a classifier failure cannot block retrieval (mirrors R1.3's "degrade to RAG on answerability check failure"). Strict 4-label `QueryType` allowlist describes query SHAPE not domain — adding a fifth label requires explicit redesign. The `_ENTITY_TOKEN_PATTERN` (`\b[A-Z][A-Z0-9_-]{2,}\b`) lives at module top here and is imported by R8.2's `failure_analysis.py` — exactly one compiled regex for entity-shape across the codebase. `RetrievalConfig.adaptive: AdaptiveRetrievalConfig` defaults `enabled=False` (zero behavioural change for existing consumers) with bounded numeric fields (`top_k_min ∈ [1, 50]` default 3, `top_k_max ∈ [1, 200]` default 15, `max_expansion_retries ∈ [0, 5]` default 2), cross-field check `top_k_min ≤ top_k_max`, and cross-config check `enabled AND use_llm_classification` requires `RetrievalConfig.enrich_lm_client` (in `RagEngine._validate_config()`). R5.1 is plumbing only — `RetrievalService.retrieve` does not yet consume the classifier; R5.2 starts dispatch on `query_type`, R5.3 escalates on `complexity` confidence.

**Dynamic top_k + task-aware weights** (R5.2, 2026-04-28) — lights up the first two consumer-facing R5 mechanisms on top of R5.1's classifier. Opt-in via `AdaptiveRetrievalConfig.enabled=True`; default `False` keeps `RetrievalService.retrieve` byte-for-byte unchanged (no classifier call, no multipliers, `trace.adaptive` stays `None`). When enabled: `RetrievalService.retrieve` calls `_compute_adaptive_params` ONCE at the top BEFORE query rewriting (the classifier sees the original user query, not LLM-generated variants — HyDE / multi-query expand a short factual question into entity-rich paraphrases that would skew classification toward COMPLEX / COMPARATIVE artificially). Complexity → effective top_k: SIMPLE → `top_k_min`, MODERATE → `base_top_k` (the static `RetrievalConfig.top_k` — promoting MODERATE to its own knob would add a third tunable for R8.3 calibration with no observable behavioural gain), COMPLEX → `top_k_max`. Query type → per-method multipliers, applied to each `method.weight` when building the parallel `weights` array for RRF fusion. `_DEFAULT_TASK_WEIGHT_PROFILES` (in `classification.py`) ships four research-informed first-cut profiles — FACTUAL vector-dominant 1.2/0.8/0.8/0.8, COMPARATIVE document/tree-dominant 0.8/1.2/0.8/1.2, ENTITY_RELATIONSHIP graph-dominant 0.8/0.8/1.5/0.8, PROCEDURAL document/tree-dominant 1.0/1.2/0.8/1.2 (vector/document/graph/tree). **Treat the numbers as defensible defaults, not tuned production values; calibrate via R8.3 benchmark before relying on them in production.** Override semantics: full replacement at the `QueryType` level (a consumer who provides only the FACTUAL profile gets defaults for the other three query types — full replacement matches "I want a custom FACTUAL profile" intent without forcing them to re-state every method); within a profile, methods absent from the dict fall back to multiplier 1.0. `RetrievalTrace.adaptive: dict[str, Any] | None` — new optional field, populated only when adaptive ran; carries `complexity`, `query_type`, `effective_top_k`, `applied_multipliers`, `classification_source`. Asserting via `trace.adaptive["applied_multipliers"]` is the supported public surface for inspecting per-method weights without reaching into service internals. `trace.timings["classification"]` records the wall-clock time the classifier took. R5.3 will add confidence-driven re-retrieval on top of the same classifier call.

**AUTO mode** (R1.4, 2026-04-28) — lights up `mode="auto"` user-facing — the recommended mode for new users. AUTO routes per query to RETRIEVAL or DIRECT based on `KnowledgeManager.get_corpus_tokens(knowledge_id)` versus `RoutingConfig.direct_context_threshold` (default 150_000 — Anthropic's tested boundary for prompt-cached long-context performance). Boundary rule: `tokens <= threshold` → DIRECT (cheaper-and-better at small sizes); otherwise → RETRIEVAL. The dataclass default stays `QueryMode.RETRIEVAL` so existing consumers are byte-for-byte unchanged. New `RagEngine._query_via_auto` does the lookup, emits `logger.info("auto routing: tokens=%d threshold=%d → %s", ...)` for operator visibility without trace overhead, and delegates to `_query_via_direct_context` or `_query_via_retrieval` (which already populate `routing_decision`). AUTO does NOT route to HYBRID by design — HYBRID adds an answerability LLM call to every query, and AUTO's job is to pick the cheapest correct strategy; consumers who want SELF-ROUTE escalation opt in with `mode="hybrid"` explicitly. AUTO does NOT add a fifth `routing_decision` enum value (R5.3 introduces the fifth value `"retrieval_then_direct"` for confidence-expansion escalation); `result.trace.routing_decision` records the chosen route (`"direct"` or `"retrieval"`), not the chosen mode. Per-query cost: ONE extra metadata-store read (`get_corpus_tokens`, typically <10 ms thanks to R1.1's per-source cache). Empty `knowledge_id` → tokens=0 → DIRECT with empty corpus (LLM responds "I don't have information"). `query_stream()` continues to refuse non-RETRIEVAL modes (AUTO included); streaming AUTO is deferred. Cost-shape comparison: RETRIEVAL ~$0.005 (best for >150K tokens), DIRECT ~$0.10 cached (best for <150K tokens), HYBRID ~$0.005-0.10 mixed (best for mixed query workloads), AUTO routes to RETRIEVAL or DIRECT (default for most users).

**Confidence expansion + LC escalation** (R5.3, 2026-04-28) — closes the R5 series with a self-healing retry loop wrapped around `_retrieve_chunks` inside `RagEngine._query_via_retrieval`. Opt-in via `AdaptiveRetrievalConfig.confidence_expansion=True` (default `False`); when disabled, `_query_via_retrieval` runs byte-for-byte unchanged. When enabled and the first attempt returns weak chunks (`max(score) < GenerationConfig.grounding_threshold`), the loop retries with `top_k *= 2` (capped at `AdaptiveRetrievalConfig.top_k_max`). After `max_expansion_retries` exhausted with chunks still weak, optional LC escalation routes to `_query_via_direct_context` when `KnowledgeManager.get_corpus_tokens(knowledge_id) <= RoutingConfig.direct_context_threshold`; otherwise the engine logs and proceeds with the last (weak) chunks. The retry loop lives at the engine layer (NOT in `RetrievalService`): the engine has access to `KnowledgeManager.get_corpus_tokens` for the LC-escalation decision and `_query_via_direct_context` for the actual escalation — service-level concerns shouldn't know about cross-strategy escalation. HYBRID is naturally excluded — `_query_via_hybrid` calls `_retrieve_chunks` directly (not `_query_via_retrieval`) — HYBRID has its own answerability check; expansion would double up. New private helpers on `RagEngine`: `_max_chunk_score(chunks) -> float | None` (returns `max(score)` or `None` for empty input) and `_should_expand(chunks, threshold) -> bool` (`max_score is None or max_score < threshold`; boundary is `<` not `<=` — a chunk score equal to the threshold is grounded, not weak). `_retrieve_chunks` gains an optional `top_k: int | None` parameter threaded through to `RetrievalService.retrieve` so the loop retries with a larger top_k without mutating the service-level default. New `RetrievalTrace.routing_decision` value `"retrieval_then_direct"` (the fifth value, distinct from plain `"direct"` because the cost shape differs — RAG-then-LC vs LC-only — and debugging consumers need to attribute escalations). New `trace.adaptive` keys when expansion runs: `expansion_attempts: int`, `expansion_outcome: "succeeded" | "exhausted_proceeded" | "exhausted_escalated_to_lc"`, `final_top_k: int`. Absent (not zero/None) when `confidence_expansion=False` — keeping "didn't run" distinct from "ran with 0 retries". Step 2 of the loop (rewriter swap) ships as a no-op placeholder — only one rewriter is configured today; future enhancement could add `AdaptiveRetrievalConfig.expansion_rewriters: list[BaseQueryRewriting]`.

**RAPTOR tree builder** (R2.2, 2026-04-29) — runtime ingestion-side foundation for RAPTOR-style hierarchical summarisation retrieval (R2.1 shipped the compile-time scaffold; R2.3 wires `RaptorRetrieval`). Opt-in via `IngestionConfig.raptor.enabled=True` (and `summary_model` set — enforced both at `RaptorConfig.__post_init__` time and defensively at builder runtime). Default-off keeps existing 1160 tests byte-for-byte unchanged: no builder constructed, no registry built, no vectors written. New `RagEngine.build_raptor_index(knowledge_id) -> RaptorBuildReport` is the consumer API; lazily constructs `RaptorTreeBuilder` + `RaptorTreeRegistry` on first call (mirrors R6.2's `_iterative_service` lazy-construction pattern) so the off path pays no construction cost. `RaptorTreeBuilder.build` runs the per-level loop: scroll leaves with `vector_role IN {"description", "raw_text", "table_row"}` (drawing components excluded by omission — their tag-style text is not amenable to narrative summarisation); cluster current level's vectors via `run_clustering` from `reasoning.modules.clustering.algorithms` (imported directly because `ClusteringService` only exposes sample-bounded `Cluster` objects, not raw per-input labels); algorithm-specific termination guard `_can_cluster_meaningfully` (K-Means: `len(current) > clusters_per_level + 1`; HDBSCAN: `len(current) >= 2 * min_cluster_size`); cap each cluster at `MAX_CLUSTER_MEMBERS_PER_SUMMARY = 20` centroid-nearest (hardcoded module-scope constant — tuning past ~30 risks summary devolving into a list); summarise each cluster concurrently through `run_concurrent` with hardcoded `_SUMMARIZE_CONCURRENCY = 5` (mirrors `IngestionConfig.analyze_concurrency` default — stays under typical Tier-1 LLM rate limits); cluster-of-one shortcut bypasses the `b.SummarizeCluster` call and reuses the single member's text with `reasoning="cluster-of-one passthrough"` (avoids burning an LLM call summarising one chunk into "a summary of one chunk"); embed via `IngestionConfig.embeddings`; persist as `vector_role="raptor_summary"` with full payload (`raptor_tree_id`, `raptor_level`, `raptor_parent_id`, `raptor_cluster_size`, `raptor_reasoning`); back-link each child point's `raptor_parent_id` via the vector store's optional `set_payload` primitive (Qdrant-native; degrades to debug-log skip when the store doesn't expose payload-only updates — back-references are an accelerator, not a correctness requirement). Atomic swap order: persist all summaries → `set_active` → GC. The order is load-bearing: if GC raises after `set_active` succeeds, the new tree is live and old summary vectors are orphans (still tagged with the old `raptor_tree_id` so retrieval correctly filters them out at query time). GC failure is logged-and-continued, not raised. The R2.2 spec accepts this orphan window as the cross-store-transactional cost; a future `gc_orphans` sweep is out of scope. `BaseVectorStore.delete(filters)` already supports payload-equality and `IN` (via list values → Qdrant `MatchAny`), so no protocol extension was needed; the GC's "delete every tree except the new one" is implemented as a two-step scroll-then-delete (existing filter shape supports equality / `IN`, not `$ne`). `RaptorBuildReport` is populated end-to-end: `level_counts` (e.g., `[1000, 100, 11, 1]`), `total_summaries` (`sum(level_counts[1:])`), `total_decompose_calls` (excludes cluster-of-one passthroughs), `total_cost_usd` (None — `LanguageModelClient` doesn't expose pricing yet), `duration_seconds`, `timings` dict with per-stage breakdown (`load_leaves`, per-level `level_N_cluster` / `level_N_summarize` / `level_N_embed` / `level_N_persist` / `level_N_parent_link`, plus `swap`, `gc`, `gc_deleted_count`). Lives at `modules/ingestion/methods/raptor/builder.py`; sibling subpackage to existing ingestion methods (NOT folded into `AnalyzedIngestionService` / `VectorIngestion`), per "RAPTOR is a distinct retrieval strategy, not a layer on top of chunk indexing".

**Multi-hop iterative retrieval** (R6, 2026-04-29) — closes the R6 phase with a complete multi-hop iterative pipeline (R6.1 config + BAML scaffold; R6.2 service + hop loop + engine arm; R6.3 post-loop DIRECT escalation). Opt-in via `IterativeRetrievalConfig.enabled=True`; default `False` keeps the existing pipeline byte-for-byte unchanged (no service constructed, no decomposer registry, no `_query_via_iterative` call). When enabled, `RagEngine._query_via_retrieval` consults the gate at the top: in `gate_mode="type"` (default) it calls R5.1's `classify_query` once and routes to `_query_via_iterative` only when the verdict is COMPLEX or ENTITY_RELATIONSHIP — gate-fail falls through to plain retrieval; in `gate_mode="llm"` it routes unconditionally and lets the first `b.DecomposeQuery` call decide via `done=true`. AUTO mode reaches `_query_via_retrieval` when it routes to RETRIEVAL, so iterative covers both RETRIEVAL and AUTO->RETRIEVAL transparently. HYBRID and DIRECT are NOT affected by design (HYBRID has its own answerability check — double-decomposing burns LLM calls without benefit; DIRECT loads the whole corpus and has no per-hop need). `IterativeRetrievalService.retrieve` (sibling subpackage to `RetrievalService`, NOT a fold-in) owns the hop loop: each iteration calls `b.DecomposeQuery` (fresh BAML registry per call, scoped to `IterativeRetrievalConfig.decomposition_model` or falling back to `RetrievalConfig.enrich_lm_client`), then `RetrievalService.retrieve(query=sub_question, knowledge_id=..., trace=True)`. Per-hop chunks merged via `_merge_chunks_dedup` (insertion order preserved; collisions update the existing slot in place — do NOT move to end; higher score wins on collision). Findings are *replaced*, not appended — the decomposer self-summarises via R6.1's prompt contract, bounding findings growth regardless of `max_hops`. Sequential by design (no `asyncio.gather` over hops — each hop depends on prior findings); within a hop, `RetrievalService` parallelism is unchanged. Termination reasons: `"done"` (decomposer verdict, also gate-fail short circuit), `"max_hops"` (loop exhausted), `"error"` (decomposer contract violation: `done=false` with empty `next_sub_question`, OR mid-loop decompose exception), plus R6.3's `"low_confidence_escalated"` and `"low_confidence_no_escalation"` (assigned only when `total_retrieve_calls > 0` — a `done` short-circuit with no retrieval is a legitimate decomposer verdict). New `IterativeOutcome(hops, termination_reason, total_decompose_calls, total_retrieve_calls)` dataclass returned from the service (exported from `rfnry_rag.retrieval`). New `RetrievalTrace.iterative_hops: list[IterativeHopTrace] | None` (default `None` — distinct from `[]` "ran with zero hops") and `iterative_termination_reason: str | None`. New `routing_decision` values `"iterative"` (R6.2) and `"iterative_then_direct"` (R6.3). `IterativeHopTrace` extended with `adaptive: dict[str, object] | None` so R5.2's per-hop classifier verdict + R5.3's expansion keys land *inside* the per-hop trace (boundary preservation; addresses R5.3's trace-data-dropped-at-boundary review pattern). Cross-config rule: `iterative.enabled=True` requires either `iterative.decomposition_model` set OR `RetrievalConfig.enrich_lm_client` set — without it, the loop would silently no-op at first decompose call; failing at `_validate_config` surfaces the misconfig at engine init. R5.3's LC escalation lives entirely at the engine layer (`_query_via_retrieval`), NOT in `RetrievalService.retrieve`, so per-hop calls naturally skip escalation — no `suppress_direct_escalation` flag needed. R6.3 post-loop DIRECT escalation: when iterative finishes naturally (`done`/`max_hops`) with weak accumulated chunks (max score < threshold) AND the corpus fits `RoutingConfig.direct_context_threshold`, the engine routes to `_query_via_direct_context` and stamps the iterative trace fields (`iterative_hops`, `iterative_termination_reason="low_confidence_escalated"`, `routing_decision="iterative_then_direct"`, plus the pre-escalation classifier verdict) onto the DIRECT result — mirrors R5.3's boundary-preservation merge to keep the multi-hop context visible. Threshold sourcing: `iterative.grounding_threshold` overrides `generation.grounding_threshold` when set; `None` (default) inherits — no hardcoded magic-number fallback. Engine-init warning (R6.3): `escalate_to_direct=True` with `RoutingConfig.direct_context_threshold` unconfigured emits a one-shot warning at `_validate_config` time — escalation will be a no-op, but consumers may run iterative without DIRECT fallback intentionally so this is not a hard error. R6.3 also lifts R5.3's `_should_expand` / `_max_chunk_score` to `retrieval/common/grounding.py` (`is_weak_chunk_signal` / `max_chunk_score`) so R5.3's retrieval arm and R6.3's iterative arm share a single source of truth for the weak-chunk boundary check; `RagEngine._should_expand` / `_max_chunk_score` remain as thin wrappers. Per-query cost: 1-`max_hops` extra LLM calls (decomposer) + 0-`max_hops` retrieval calls + 0-1 DIRECT escalation calls; gate keeps cheap queries on the cheap path.

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
- 1118 tests total across both SDKs (Phase F adds +21 vs. the Phase E baseline of 997: F2 DrawingIngestionService status-transition + re-entry tests, F3.1 DXF TEXT/MTEXT off-page connectors, F3.2 DXF paperspace layout rendering, F3.5 Gemini vision provider; R4 adds +9 chunk-ordering unit cases; R3 adds +8 document-expansion unit cases; R8.1 adds +9 retrieval-trace unit cases; R8.2 adds +8 failure-classification unit cases; R8.3 adds +8 benchmark-harness unit + CLI cases (6 unit + 1 CLI + 1 LLM-judge unit); R1.1 adds +6 corpus-token-counting unit cases; R1.2 adds +7 routing/DIRECT-mode unit cases; R1.3 adds +8 routing/HYBRID-mode unit cases; R1.4 adds +6 routing/AUTO-mode unit cases and removes the R1.2 AUTO-raises regression guard for net +5; cumulative R5 contribution +27: R5.1 adds +9 (8 main + 1 polish registry-guard regression test); R5.2 adds +9 (7 main + 2 polish override-semantics + tree-multiplier regression tests); R5.3 adds +9 (8 main + 1 polish adaptive-merge regression test); cumulative R6 contribution +27: R6.1 adds +8 iterative-config unit cases; R6.2 adds +13 iterative service + hop loop + engine arm cases (12 main + 1 engine-init validation guard); R6.3 adds +6 post-loop escalation cases; cumulative R2 contribution to date +33: R2.1 adds +13 RAPTOR config + registry tests; R2.2 adds +20 builder + atomic-swap + GC tests)

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
- `RoutingConfig.mode`: `QueryMode` enum, default `RETRIEVAL` (backward compat); `DIRECT` is live in R1.2, `HYBRID` is live in R1.3 (requires `hybrid_answerability_model`), `AUTO` is live in R1.4 (recommended for new users)
- `RoutingConfig.direct_context_threshold`: `1_000 ≤ n ≤ 2_000_000`, default 150_000 (Anthropic's tested boundary for prompt-cached long-context performance; AUTO routes `tokens <= threshold` to DIRECT)
- `AdaptiveRetrievalConfig` (R5.1): `enabled` defaults False (plumbing only); `top_k_min` `1 ≤ n ≤ 50` default 3; `top_k_max` `1 ≤ n ≤ 200` default 15; `top_k_min ≤ top_k_max` enforced in `__post_init__`; `max_expansion_retries` `0 ≤ n ≤ 5` default 2 (consumed by R5.3 confidence expansion); `confidence_expansion` defaults False (R5.3 opt-in); `use_llm_classification` cross-config requires `RetrievalConfig.enrich_lm_client` (validated in `RagEngine._validate_config`); `task_weight_profiles` is unbounded — consumer-supplied per-method overrides normalised by fusion at consume time
- `IterativeRetrievalConfig` (R6 — R6.1 scaffold; R6.2 hop loop + engine arm; R6.3 post-loop DIRECT escalation): `enabled` defaults False (byte-for-byte unchanged when off); `max_hops` `1 ≤ n ≤ 10` default 3; `gate_mode` allowlist `{"type", "llm"}` default `"type"`; `escalate_to_direct` defaults True; `grounding_threshold` `0.0 ≤ t ≤ 1.0` when set, default `None` (inherits `GenerationConfig.grounding_threshold` at consume time — no hardcoded magic-number fallback); cross-field rule (in `__post_init__`): `enabled=True AND gate_mode="llm"` requires `decomposition_model`; cross-config rule (in `RagEngine._validate_config`, R6.2): `enabled=True` requires either `decomposition_model` or `RetrievalConfig.enrich_lm_client` set — without an LLM client the loop would silently no-op on the first `DecomposeQuery` call. R6.3 init-time warning: `escalate_to_direct=True` with `RoutingConfig.direct_context_threshold` unconfigured logs once at init (escalation is a no-op but not a hard error — consumers may intentionally run iterative without DIRECT fallback). Lives at `modules/retrieval/iterative/config.py`; sibling subpackage to `RetrievalService`, not folded in. Raises `ConfigurationError` on validation failures (harmonized with R5.1's `AdaptiveRetrievalConfig`).
- `RaptorConfig` (R2 — R2.1 scaffold; R2.2 builder + R2.3 retrieval method follow): `enabled` defaults False (byte-for-byte unchanged when off, no consumer reads the fields yet); `max_levels` `1 ≤ n ≤ 10` default 3; `cluster_algorithm` allowlist `{"kmeans", "hdbscan"}` default `"kmeans"`; `clusters_per_level` `2 ≤ n ≤ 100` default 10; `min_cluster_size` `2 ≤ n ≤ 100` default 5; `summary_max_tokens` `50 ≤ n ≤ 2000` default 256; `summary_model: LanguageModelClient | None = None`. Cross-field rule (in `__post_init__`): `enabled=True` requires `summary_model` (without it, the builder would silently no-op at first `SummarizeCluster` call). Lives at `modules/ingestion/methods/raptor/config.py`; sibling subpackage to existing ingestion methods (NOT folded into `AnalyzedIngestionService` / `VectorIngestion`), per "RAPTOR is a distinct retrieval strategy, not a layer on top of chunk indexing". Raises `ConfigurationError` on validation failures.
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
