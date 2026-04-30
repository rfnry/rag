# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`rfnry-rag` is a Python retrieval toolkit. One SDK, one purpose: ingest documents, build modular retrieval pipelines (vector + document + graph), route between indexed retrieval and full-context generation based on corpus size, and stay out of the LLM's way.

## Development philosophy

The toolkit is designed around a single test: **does this feature compose with smarter models, or compete with them?** Anything that competes is on a deprecation clock. When deciding what to add, change, or remove, apply the same test:

- **Keep:** infrastructure the model never touches (async I/O, error isolation, protocols, structured outputs), structured input that better models exploit better (graph emission, document expansion at index time), and code that gets out of the way as the model gets stronger (`AUTO` routing, `FULL_CONTEXT` mode).
- **Strip:** anything that exists to compensate for old model limitations — small context windows, weak instruction-following, unreliable structured outputs. These features age out and become maintenance debt.
- **Don't add abstractions for hypothetical use cases.** Three similar lines beats a premature framework. No "in case the model gets worse" guardrails.

Concretely: do not add LLM query classifiers, multi-hop decomposers, hierarchical summarization, fuzzy-match cross-page linkers, or any other workaround for capabilities the model already has or will soon. If a feature is "the model can't do X, so we do X for it," it does not belong here.

### Pre-change checklist

**Before proposing or implementing any feature, change, or refactor, answer these seven questions in your response.** Mandatory; no exceptions, including for "small" changes. Any "no" or "yes — because the current model is weak at X" means the change does not ship; rework it.

1. **Will this still be wanted if the next-gen model has 10× the context window?** *(If the design assumes context scarcity — hierarchical summarization passes, multi-hop query decomposition, aggressive top-k pruning the model could absorb directly — answer is no. The whole `INDEXED` vs `FULL_CONTEXT` axis exists because we expect this number to grow.)*
2. **Will this still be wanted if the next-gen model has noticeably stronger reasoning and calibration?** *(If the design second-guesses the model — LLM query classifiers picking top_k, adaptive weight profiles, confidence-expansion retries, failure-type taxonomies — answer is no.)*
3. **Does this feature get *more* useful as the model improves, or *less*?** *(More = substrate / compounds: better embeddings → better vector search; better vision → richer drawing extraction; bigger context → more corpora hit `FULL_CONTEXT`. Less = anti-survives. Name which.)*
4. **If "less," is it a structural boundary independent of model strength?** *(Examples that survive: BAML schema typing, prompt-injection fencing, domain-vocabulary allowlists, public-input length caps, per-method error isolation, the grounding gate, the benchmark harness, the trace. If yes, keep. If no, cut.)*
5. **Does this feature exist because the current model is weak at X?** *(Be specific: small context window, weak cross-document reasoning, overconfidence on irrelevant chunks, brittle structured output, weak instruction-following on long prompts, poor cross-page synthesis on diagrams. If yes, the design is wrong — rework it to give the model better input and let it do the reasoning, not to do the reasoning ourselves.)*
6. **Does the trace + benchmark harness already surface the failure mode this feature is meant to prevent?** *(If yes, the feature is redundant — the eval is the eval. Add cases to the harness instead of adding code to the pipeline.)*
7. **Would I still want this if the model were two generations ahead?** *(Forward-only. If the answer is "no, the next model handles this natively at query time," the change does not ship. This is how RAPTOR, multi-hop iterative retrieval, fuzzy cross-page linkers, and HYBRID routing got cut.)*

State the answers explicitly. If you cannot give a clean answer to a question, the design is not ready and you may not implement it yet.

## Code style

- **Default to writing no comments.** Well-named identifiers, dataclasses, and protocols already say what the code does. Only add a comment when the *why* is non-obvious — a hidden constraint, a subtle invariant, a workaround for a specific external bug. If removing the comment wouldn't confuse a future reader, don't write it.
- **Never explain what the code does.** No `# Set up the client`. No multi-paragraph docstrings on simple functions. One short docstring line maximum on public APIs; otherwise nothing.
- **No "added for X / used by Y" references.** That information rots and belongs in commit messages, not source.
- **No backwards-compat shims, deprecation comments, or `// removed` markers.** Delete fully; rely on git history.
- **No defensive validation past system boundaries.** Trust internal callers and framework guarantees. Validate at user-input/external-API boundaries only.
- **No half-finished implementations.** No TODOs in shipped code. Either finish the thing or don't add it.

## Commands

All tasks run via [poethepoet](https://github.com/nat-n/poethepoet). Prefix with `uv run` if not in the venv:

```bash
poe format            # ruff format
poe check             # ruff lint
poe check:fix         # ruff lint with auto-fix
poe typecheck         # mypy src/
poe test              # pytest (asyncio_mode=auto, pythonpath=src)
poe test:cov          # pytest with coverage
poe baml:generate     # regenerate BAML clients
```

Run a single test: `pytest tests/retrieval/test_methods.py::test_name -v`

## Architecture

### Package structure

```
src/rfnry_rag/
├── __init__.py          # top-level re-exports
├── server.py            # RagEngine — async context manager, dynamic pipeline assembly
├── logging.py           # get_logger, query_logging_enabled
├── concurrency.py       # run_concurrent
├── exceptions/          # one file per error family
├── providers/           # LanguageModelClient, LanguageModel, registry, facades
├── models/              # Source, Chunk, vector DTOs, retrieval results, stats
├── config/              # all config dataclasses, one place
├── ingestion/           # base + service + chunk/ + methods/ + drawing/
├── retrieval/           # base + service + fusion + methods/ + reranking + routing
├── generation/          # service + grounding + formatting + models
├── stores/              # vector/ + document/ + graph/ + metadata/
├── knowledge/           # KnowledgeManager (CRUD + corpus-token accounting for AUTO routing)
├── observability/       # trace + benchmark + metrics
├── cli/                 # commands + helpers + config loader + output formatters
└── baml/                # baml_src/ + baml_client/ + version_check
```

### Entry points

- **SDK:** `RagEngine` in `server.py` — async context manager. `async with RagEngine(config) as rag:`
- **CLI:** `rfnry-rag <cmd>` (ingest, query, benchmark, knowledge).
- **Imports:** `from rfnry_rag import RagEngine`. Errors via `from rfnry_rag.exceptions import RagError, IngestionError, ...`. Domain types via `from rfnry_rag.models import Source, Chunk, RetrievedChunk`.

### Retrieval pipeline

`RagEngine.query()` runs:

1. **Routing.** `RoutingConfig.mode` selects between `INDEXED` (the standard pipeline below), `FULL_CONTEXT` (load the corpus into a prompt-cached prefix, skip retrieval), and `AUTO` (per-query corpus-token threshold dispatches between the two via `KnowledgeManager.get_corpus_tokens`).
2. **Multi-path retrieval.** Configured methods run concurrently; results merge via reciprocal rank fusion with per-method weights:
   - `VectorRetrieval` — dense + BM25 fused internally.
   - `DocumentRetrieval` — Postgres FTS + substring (requires document store).
   - `GraphRetrieval` — entity lookup + N-hop traversal (requires graph store).
3. **Reranking** (optional). Cross-encoder against the original query (Cohere, Voyage).
4. **Generation.** Grounding gate → context assembly via `chunks_to_context()` (`SCORE_DESCENDING` default; `PRIMACY_RECENCY` and `SANDWICH` opt-in) → LLM generation.

Methods carry `weight` and `top_k` configuration. Per-method error isolation: catch, log, continue. Failure of one path does not break others.

### Optional trace

Pass `trace=True` to `RagEngine.query()` to receive a `RetrievalTrace` (in `observability/trace.py`) capturing per-stage state: `query`, `per_method_results` (keyed by `BaseRetrievalMethod.name`, includes empty-result methods), `fused_results`, `reranked_results`, `final_results`, `grounding_decision`, `routing_decision`, `timings`, `knowledge_id`. Default `trace=False` is byte-for-byte unchanged. The `None` vs `[]` distinction is load-bearing: `reranked_results is None` means "reranker not configured", `[]` means "ran with no input". `query_stream` does not collect a trace.

### Drawing ingestion

For diagram-first documents (schematics, P&ID, wiring, mechanical drawings) `DrawingIngestionService` runs `render → extract → ingest`:

- **PDF pages** → vision LLM produces structured per-page output (components, labels, off-page connector tags) via `AnalyzeDrawingPage`.
- **DXF files** → `ezdxf` native parse over modelspace plus all paperspace layouts in tab order — one ingested page per layout, no LLM calls.
- **Cross-page references** are emitted into the graph store as edge candidates (off-page connector tags, label references). The model resolves cross-sheet connectivity at query time over the assembled graph; the toolkit does not pre-link pages.

Symbol vocabularies are consumer-configurable via `DrawingIngestionConfig` (ships IEC 60617 + ISA 5.1 defaults).

### Benchmark harness

`RagEngine.benchmark(cases) -> BenchmarkReport` (and CLI `rfnry-rag benchmark cases.json -k <knowledge_id>`) aggregates `ExactMatch`, `F1Score`, `LLMJudgment`, `RetrievalRecall`, `RetrievalPrecision` across cases, with per-case traces in the report. `retrieval_recall` / `retrieval_precision` are `None` when at least one case omits `expected_source_ids` (N/A is distinct from 0.0). Failure rule: F1 < `failure_threshold` (default 0.5) OR `trace.grounding_decision == "ungrounded"`.

### Modular pipeline

Retrieval and ingestion are protocol-based plugin architectures. No mandatory vector DB or embeddings — at least one retrieval path (vector, document, or graph) must be configured.

- **`BaseRetrievalMethod` / `BaseIngestionMethod`** — protocol interfaces in `retrieval/base.py` and `ingestion/base.py`.
- **Method classes** — `VectorRetrieval`, `DocumentRetrieval`, `GraphRetrieval` (retrieval); `VectorIngestion`, `DocumentIngestion`, `GraphIngestion`, `AnalyzedIngestion`, `DrawingIngestion` (ingestion). Each is self-contained with error isolation and timing logs.
- **Dynamic assembly** — `RagEngine.initialize()` builds method lists from config, validates cross-config constraints, assembles `RetrievalService` and `IngestionService` with method-list dispatch.
- **`BaseIngestionMethod.required: bool`** is part of the protocol. `VectorIngestion` and `DocumentIngestion` default `required=True`; `GraphIngestion` defaults `required=False`. Required-method failures abort the ingest with `IngestionError` and skip the metadata commit.
- **Graph ingestion is consumer-agnostic by default.** `GraphIngestionConfig` lets consumers supply their own entity-type regex patterns, relationship keyword map, and fallback edge type. Empty config → type inference falls through to `DiscoveredEntity.category.lower()`; cross-references with no keyword match become generic `MENTIONS` edges.

### Error hierarchy

```
RagError (root, catch-all for SDK errors)
├── ConfigurationError
├── IngestionError
│   ├── ParseError
│   ├── EmptyDocumentError
│   ├── EmbeddingError
│   └── IngestionInterruptedError
├── RetrievalError
├── GenerationError
├── StoreError
│   ├── DuplicateSourceError
│   └── SourceNotFoundError
└── InputError(RagError, ValueError)
```

`RagError` is the root — there is no separate `SdkBaseError`. Catch the specific subclasses, or `RagError` for the catch-all.

### LLM integration

All LLM calls go through BAML for structured output parsing, retry/fallback, and observability. Edit `baml/baml_src/`; regenerate with `poe baml:generate`. Never edit `baml_client/`.

`LanguageModelClient` (in `providers/client.py`) builds a BAML `ClientRegistry` with primary + optional fallback provider routing. `LanguageModel` (in `providers/provider.py`) configures a single endpoint (API key, base URL, model). Facades (`Embeddings`, `Vision`, `Reranking` in `providers/facades.py`) dispatch to the correct backend at runtime based on the configured provider.

### When to use BAML for a new feature

BAML's value is **structured-output parsing** with primary/fallback routing. After the 2026-04 prune the SDK keeps only 7 BAML functions, all substrate-only (vision extraction, entity extraction, index-time synthetic-query generation, answer-quality judging, relevance gating, cross-page synthesis).

Before adding a new BAML function, answer all 5. **Two or more "no" → don't use BAML.**

1. **Does the caller need a typed object, not a string?** If the consumer immediately stringifies the output or treats it as free text, BAML's structured-output value is wasted. *Substrate check: does code downstream of the call read named fields?*

2. **Is the schema a system boundary?** Does the parsed output flow into a store / index / mapper that requires specific shapes (`DiscoveredEntity`, `DetectedComponent`, `AnswerQualityJudgment`)? *If the schema only exists to feed the next prompt, it's not a boundary — just emit text.*

3. **Will the caller get *more* useful as the model improves?** If better reasoning makes the call redundant (e.g. "decide if we should retrieve" vanishes when the model handles irrelevant context natively), it fails the substrate test from this file's "Pre-change checklist" and the BAML wrapping is a deprecation magnet.

4. **Does the value justify the friction tax?** Adding a BAML function means: a `.baml` source file, a `poe baml:generate` regen step, a `baml_client/` diff, a `ClientRegistry` plumbing call, and three contract tests touching it. *For one-off structured output, native SDK JSON-mode or tool calling may be lighter.*

5. **Is this index-time augmentation, not query-time decision-making?** Index-time use (chunk expansion, entity extraction, vision OCR) compounds with model improvement. Query-time LLM-as-router/classifier/decomposer competes with the model. *Index-time → BAML is fine. Query-time → suspect.*

**Decision rule:**
- Five yeses → use BAML.
- One no on #4 only → use BAML, but consider native JSON-mode first.
- Two or more nos → don't use BAML. Either skip the structured shape (return text), or make it a Python-side regex / dataclass parse.

**Removed BAML functions (do not add back without satisfying the checklist):** `JudgeRetrievalNecessity`, `GenerateReasoningStep`, `CompressRetrievedContext`, `AnalyzeQuery`, `GenerateQueryVariants`, `RerankChunks`, `GenerateAnswer` (replaced by native `LanguageModelClient.generate_text`). See `docs/plans/2026-04-30-baml-prune.md` for the per-function 7-question rationale.

## Key patterns

- **Protocol-based abstraction.** No inheritance; `Protocol` classes define interfaces (`BaseEmbeddings`, `BaseRetrievalMethod`, `BaseIngestionMethod`, `BaseReranking`, etc.). Any conforming object works.
- **Facade pattern.** `Embeddings(LanguageModel)`, `Vision(LanguageModel)`, `Reranking(LanguageModel | LanguageModelClient)` are public facades that select the correct private provider implementation at runtime.
- **Modular pipeline.** Services receive `list[BaseRetrievalMethod]` / `list[BaseIngestionMethod]` and dispatch generically. Per-method error isolation.
- **Async-first.** All I/O is async. Services use `async def`; stores use asyncpg / aiosqlite.
- **Service pattern.** Each module has a `Service` class with dependencies injected via `__init__`.
- **Domain-neutral by default.** No hardcoded domain vocabulary in BAML prompts. Features needing vocabulary expose consumer-overridable config with empty defaults; values are validated against an allowlist where applicable.

## Contract tests

These act as regression guards — they enforce whole-class invariants:

- `test_baml_prompt_fence_contract.py` — every user-controlled BAML prompt parameter must be fenced. Fails if any new `.baml` file introduces an unfenced interpolation.
- `test_baml_prompt_domain_agnostic.py` — scans BAML sources for domain-bias vocabulary; fails on any banned term.
- `test_config_bounds_contract.py` — every `int` / `float` field in a config dataclass must have a `__post_init__` bounds check or carry a `# unbounded: <reason>` marker.

## Linting & style

- Ruff: line-length 120, target py312, rules: E, F, I, UP, B, SIM, RUF022.
- MyPy: python 3.12, ignores missing imports.
- Both tools exclude `baml_client/`.

## Testing

- pytest with `asyncio_mode = "auto"` — no `@pytest.mark.asyncio` needed.
- Tests use `AsyncMock` and `SimpleNamespace` for lightweight mocking.
- Tests live in `tests/`, mirroring source layout (`tests/ingestion/`, `tests/retrieval/`, `tests/generation/`, `tests/stores/`, `tests/observability/`, `tests/contracts/`).

## Config defaults and enforced bounds

`__post_init__` validators reject pathological values at construction time:

- `IngestionConfig.chunk_size_unit`: `Literal["chars", "tokens"]`, default `"tokens"`. Default `chunk_size=375` tokens, `chunk_overlap=40`.
- `IngestionConfig.parent_chunk_size`: sentinel `-1` (default) resolves to `3 * chunk_size`; explicit `0` disables parent-child indexing.
- `IngestionConfig.document_expansion`: nested `DocumentExpansionConfig`. Defaults disabled. When `enabled=True`, `lm_client` is required.
- `AnalyzedIngestion.dpi`: `72 ≤ dpi ≤ 600`, default 300.
- `AnalyzedIngestion.analyze_concurrency`: `1 ≤ n ≤ 100`, default 5.
- `AnalyzedIngestion.analyze_text_skip_threshold_chars`: `0 ≤ n ≤ 100_000`, default 300.
- `RetrievalConfig.top_k`: `1 ≤ top_k ≤ 200`.
- `VectorRetrieval.bm25_max_chunks`: `≤ 200_000`.
- `VectorRetrieval.bm25_max_indexes`: `1 ≤ n ≤ 1000`, default 16.
- `RoutingConfig.mode`: `QueryMode` enum, default `INDEXED`. Other values: `FULL_CONTEXT`, `AUTO`.
- `RoutingConfig.full_context_threshold`: `1_000 ≤ n ≤ 2_000_000`, default 150_000 (AUTO routes corpora `≤ threshold` to `FULL_CONTEXT`). Default = Anthropic's ~200k stuff-it-all anchor minus ~25% headroom for system prompt + history + question + answer; do **not** raise to match a model's advertised window (Lost-in-the-Middle / LaRA: effective ≪ advertised).
- `GenerationConfig`: `grounding_enabled=True` requires `grounding_threshold > 0` and an `lm_client`.
- `GenerationConfig.chunk_ordering`: `ChunkOrdering` enum, default `SCORE_DESCENDING`.
- `BenchmarkConfig.concurrency`: `1 ≤ n ≤ 20`, default 1.
- `BenchmarkConfig.failure_threshold`: `0.0 ≤ t ≤ 1.0`, default 0.5.
- `DrawingIngestionConfig.dpi`: `150 ≤ dpi ≤ 600`, default 400.
- `DrawingIngestionConfig.analyze_concurrency`: `1 ≤ n ≤ 100`, default 5.
- `DrawingIngestionConfig.relation_vocabulary`: every target must be in `ALLOWED_RELATION_TYPES`.
- `GraphIngestionConfig.entity_type_patterns`: regex strings compiled at `__post_init__`.
- `GraphIngestionConfig.relationship_keyword_map`: all values must be in `ALLOWED_RELATION_TYPES`.
- `LanguageModelClient.timeout_seconds`: `> 0`, default 60.
- `LanguageModelClient.temperature`: `0.0 ≤ t ≤ 2.0`.
- `LanguageModel.context_size`: `int | None`, default `None`. When set, must be `≥ 1`; declares the model's advertised input window. Used as a *safety cap*, not a routing threshold: `RagEngine.initialize()` refuses configs where `RoutingConfig.full_context_threshold + 16_000 (non-output reserve) + LanguageModelClient.max_tokens (output reserve)` exceeds it.
- `Neo4jGraphStore.password`: required.
- Public-input bounds: query ≤ 32 000 chars, `ingest_text` ≤ 5 000 000 chars, metadata ≤ 50 keys × 8 000 chars.

## Environment variables

- `RFNRY_RAG_LOG_ENABLED=true` / `RFNRY_RAG_LOG_LEVEL=DEBUG` — SDK logging.
- `RFNRY_RAG_LOG_QUERIES=true` — include raw query text in logs (off by default; PII-safe). Use `rfnry_rag.logging.query_logging_enabled()` when adding new query-logging sites.
- `RFNRY_RAG_BAML_LOG=info|warn|debug` — BAML runtime logging (SDK sets `BAML_LOG` from this).
- `BAML_LOG=info|warn|debug` — BAML runtime logging (direct override).
- `BOUNDARY_API_KEY` — Boundary collector key, process-global.
- Config lives at `~/.config/rfnry_rag/config.toml` + `.env`.
