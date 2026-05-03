# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`rfnry-knowledge` is a **provider-agnostic** Python retrieval engine. One SDK, one purpose: ingest documents, build modular retrieval pipelines (vector + document + graph), route between indexed retrieval and full-context generation based on corpus size, and stay out of the LLM's way. The library defines contracts; the consumer brings provider implementations and plugs them in.

## Development philosophy

The toolkit is designed around a single test: **does this feature compose with smarter models, or compete with them?** Anything that competes is on a deprecation clock. When deciding what to add, change, or remove, apply the same test:

- **Keep:** infrastructure the model never touches (async I/O, error isolation, protocols, structured outputs), structured input that better models exploit better (graph emission, document expansion at index time), and code that gets out of the way as the model gets stronger (`AUTO` routing, `FULL_CONTEXT` mode).
- **Strip:** anything that exists to compensate for old model limitations — small context windows, weak instruction-following, unreliable structured outputs. These features age out and become maintenance debt.
- **Don't add abstractions for hypothetical use cases.** Three similar lines beats a premature framework. No "in case the model gets worse" guardrails.

Concretely: do not add LLM query classifiers, multi-hop decomposers, hierarchical summarization, fuzzy-match cross-page linkers, or any other workaround for capabilities the model already has or will soon. If a feature is "the model can't do X, so we do X for it," it does not belong here.

### Provider-agnostic invariant

The library imports **zero** vendor SDKs. There is no `anthropic` / `openai` / `google-genai` / `voyageai` / `cohere` / `tiktoken` / `fastembed` dependency. All vendor-aware code lives in BAML (which itself is provider-aware via consumer-supplied `ClientRegistry`) or in consumer-supplied Protocol implementations. If you find yourself adding `import openai` or `isinstance(x, AnthropicSomething)`, the design is wrong — rework it to accept a Protocol the consumer satisfies, or route the call through BAML.

The library is also decoupled from `rfnry-protocols` and `rfnry-providers`: no imports from either.

### Pre-change checklist

**Before proposing or implementing any feature, change, or refactor, answer these seven questions in your response.** Mandatory; no exceptions, including for "small" changes. Any "no" or "yes — because the current model is weak at X" means the change does not ship; rework it.

1. **Will this still be wanted if the next-gen model has 10× the context window?** *(If the design assumes context scarcity — hierarchical summarization passes, multi-hop query decomposition, aggressive top-k pruning the model could absorb directly — answer is no.)*
2. **Will this still be wanted if the next-gen model has noticeably stronger reasoning and calibration?** *(If the design second-guesses the model — LLM query classifiers picking top_k, adaptive weight profiles, confidence-expansion retries, failure-type taxonomies — answer is no.)*
3. **Does this feature get *more* useful as the model improves, or *less*?** *(More = substrate / compounds. Less = anti-survives. Name which.)*
4. **If "less," is it a structural boundary independent of model strength?** *(Examples that survive: BAML schema typing, prompt-injection fencing, public-input length caps, per-method error isolation, the grounding gate, the benchmark harness, the trace.)*
5. **Does this feature exist because the current model is weak at X?** *(If yes, the design is wrong — rework it.)*
6. **Does the trace + benchmark harness already surface the failure mode this feature is meant to prevent?** *(If yes, the feature is redundant — the eval is the eval.)*
7. **Would I still want this if the model were two generations ahead?** *(Forward-only.)*

State the answers explicitly. If you cannot give a clean answer to a question, the design is not ready.

## Code style

- **Default to writing no comments.** Well-named identifiers, dataclasses, and protocols already say what the code does. Only add a comment when the *why* is non-obvious.
- **Never explain what the code does.** No `# Set up the client`. No multi-paragraph docstrings on simple functions.
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
src/rfnry_knowledge/
├── __init__.py              # top-level re-exports
├── knowledge/engine.py      # KnowledgeEngine — async context manager, dynamic pipeline assembly
├── common/logging.py        # get_logger, query_logging_enabled
├── common/concurrency.py    # run_concurrent
├── exceptions/              # one file per error family
├── providers/               # ProviderClient, BaseEmbeddings/BaseReranking/etc Protocols, BAML registry, TokenUsage
├── models/                  # Source, Chunk, vector DTOs, retrieval results, stats
├── config/                  # all config dataclasses, one place
├── ingestion/               # base + service + chunk/ + methods/ + drawing/
├── retrieval/               # base + service + fusion + methods/{semantic,keyword,entity} + reranking/ (Protocol only)
├── generation/              # service + grounding + formatting + models
├── stores/                  # vector/ + document/ + graph/ + metadata/
├── observability/           # trace + benchmark + metrics
├── telemetry/               # QueryTelemetryRow, IngestTelemetryRow, sinks, context, BAML usage normalization
└── baml/                    # baml_src/ + baml_client/ + version_check
```

No `cli/` directory. The library is SDK-only; the host owns transport.

### Entry points

- **SDK:** `KnowledgeEngine` in `knowledge/engine.py` — async context manager. `async with KnowledgeEngine(config) as engine:`
- **Imports:** `from rfnry_knowledge import KnowledgeEngine`. Errors via `from rfnry_knowledge.exceptions import KnowledgeEngineError, IngestionError, ...`. Domain types via `from rfnry_knowledge.models import Source, Chunk, RetrievedChunk`.

### Provider contract

- **`ProviderClient`** (in `providers/provider.py`) — a frozen dataclass: `name`, `model`, `api_key: SecretStr`, `options: dict`, plus `max_retries`, `max_tokens`, `temperature`, `timeout_seconds`, `context_size`, optional `fallback`, `strategy`. The `name` is a free-form string the consumer chooses; BAML uses it as the `provider` key on `ClientRegistry.add_llm_client`. The library never branches on `name` and never imports vendor SDKs.

- **`BaseEmbeddings` / `BaseSparseEmbeddings` / `BaseReranking` / `TokenCounter`** (in `providers/protocols.py`) — Protocols. Consumer implements; engine calls. `BaseEmbeddings.embed` returns `EmbeddingResult(vectors, usage)`; `BaseReranking.rerank` returns `RerankResult(chunks, usage)`. `usage: TokenUsage | None` mirrors the rfnry agent SDK shape (`input` / `output` / `cache_creation` / `cache_read` keys, missing → 0).

- **`build_registry(ProviderClient)`** in `providers/registry.py` — builds a BAML `ClientRegistry` from a `ProviderClient`; honors `strategy="fallback"` by adding both clients and a router.

- **`generate_text(client, system, history, user)` / `stream_text(...)`** in `providers/text_generation.py` — BAML-backed text generation through the consumer-supplied `ProviderClient`. No vendor SDK imports.

### Retrieval pipeline

The library is organized around three peer pillars — **Semantic**, **Keyword**, **Entity** — that run in parallel inside `QueryMode.RETRIEVAL` and merge via reciprocal rank fusion. RAG is one approach among many; the engine itself doesn't gravitate around it.

`KnowledgeEngine.query()` runs:

0. **Index-time enrichment** (ingestion-side, not query-time). Chunks may be enriched at ingest time via `chunk_context_headers`, `DocumentExpansionConfig` (BAML `GenerateSyntheticQueries`), and `ContextualChunkConfig` (BAML `SituateChunk`). Composes orthogonally; consumers opt in.
1. **Routing.** `RoutingConfig.mode` selects between `RETRIEVAL`, `DIRECT`, and `AUTO` (per-query corpus-token threshold via `KnowledgeManager.get_corpus_tokens`).
2. **Three-pillar retrieval.** Configured methods run concurrently; results merge via reciprocal rank fusion with per-method weights:
   - `SemanticRetrieval` — dense embeddings + optional sparse hybrid (vector store).
   - `KeywordRetrieval` — `backend="bm25"` (in-memory over the vector store) or `backend="postgres_fts"` (Postgres FTS + substring on the document store).
   - `EntityRetrieval` — entity lookup + N-hop traversal (graph store).
3. **Reranking** (optional). Cross-encoder against the original query. Consumer-supplied `BaseReranking` impl.
4. **Generation.** Grounding gate (BAML `CheckRelevance`) → context assembly → BAML `GenerateText`.

Methods carry `weight` and `top_k` configuration. Per-method error isolation: catch, log, continue.

### Optional trace

Pass `trace=True` to `KnowledgeEngine.query()` to receive a `RetrievalTrace` capturing per-stage state. Default `trace=False` is byte-for-byte unchanged.

### Drawing ingestion

For diagram-first documents `DrawingIngestionService` runs `render → extract → ingest`:

- **PDF pages** → BAML `AnalyzeDrawingPage` produces structured per-page output (components, labels, off-page connector tags) via the consumer's `ProviderClient`.
- **DXF files** → `ezdxf` native parse over modelspace plus all paperspace layouts in tab order — one ingested page per layout, no LLM calls.
- **Cross-page references** are emitted into the graph store as edge candidates. The model resolves cross-sheet connectivity at query time over the assembled graph.

Symbol vocabularies are consumer-configurable via `DrawingIngestionConfig`.

### Benchmark harness

`KnowledgeEngine.benchmark(cases) -> BenchmarkReport` aggregates `ExactMatch`, `F1Score`, `LLMJudgment`, `RetrievalRecall`, `RetrievalPrecision` across cases, with per-case traces. `LLMJudgment` takes a `ProviderClient` for BAML routing.

### Modular pipeline

Retrieval and ingestion are protocol-based plugin architectures. No mandatory vector DB or embeddings — at least one retrieval pillar (`SemanticRetrieval`, `KeywordRetrieval`, or `EntityRetrieval`) must be configured.

- **`BaseRetrievalMethod` / `BaseIngestionMethod`** — protocol interfaces in `retrieval/base.py` and `ingestion/base.py`.
- **Method classes** — `SemanticRetrieval`, `KeywordRetrieval`, `EntityRetrieval` (retrieval); `SemanticIngestion`, `KeywordIngestion`, `EntityIngestion`, `AnalyzedIngestion`, `DrawingIngestion` (ingestion). Each is self-contained with error isolation and timing logs.
- **Dynamic assembly** — `KnowledgeEngine.initialize()` builds method lists from config, validates cross-config constraints, assembles `RetrievalService` and `IngestionService`.
- **`BaseIngestionMethod.required: bool`** is part of the protocol. Required-method failures abort the ingest with `IngestionError`.
- **Entity ingestion is consumer-agnostic by default.** `EntityIngestionConfig` lets consumers supply their own entity-type regex patterns, relationship keyword map, and fallback edge type.

### Error hierarchy

```
KnowledgeEngineError (root, catch-all for SDK errors)
├── ConfigurationError
├── IngestionError
│   ├── ParseError
│   ├── EmptyDocumentError
│   ├── EmbeddingError
│   └── IngestionInterruptedError
├── EnrichmentSkipped
├── RetrievalError
├── GenerationError
├── StoreError
│   ├── DuplicateSourceError
│   └── SourceNotFoundError
└── InputError(KnowledgeEngineError, ValueError)
```

`KnowledgeEngineError` is the root. Catch the specific subclasses, or `KnowledgeEngineError` for the catch-all.

### Observability + Telemetry

Two always-on first-class modules on `KnowledgeEngineConfig`:

- **`rfnry_knowledge.observability.Observability`** — qualitative event stream. `await obs.emit(level, kind, message, **context)` builds an `ObservabilityRecord` (Pydantic) and dispatches via the configured `Sink`. Default sink is `JsonlStderrSink`. Library-defined `kind` vocabulary: `query.start`/`query.success`/`query.refused`/`query.error`, `ingest.start`/`ingest.success`/`ingest.partial`/`ingest.error`, `provider.call`/`provider.error`, `retrieval.method.success`/`retrieval.method.error`, `ingestion.method.success`/`ingestion.method.error`, `enrichment.skipped`, `vision.page.skipped`, `routing.decision`. Adding a kind is non-breaking; renaming/removing is breaking.
- **`rfnry_knowledge.telemetry.Telemetry`** — row-per-transaction. Two row types: `QueryTelemetryRow`, `IngestTelemetryRow` (Pydantic). One row written per `KnowledgeEngine.query()` / `ingest()` / `ingest_text()` invocation; outcome / duration / token totals / per-method timings populate. Default sink `JsonlStderrSink`; `SqlAlchemyTelemetrySink` persists rows via `SQLAlchemyMetadataStore` (tables `knowledge_query_telemetry`, `knowledge_ingest_telemetry` auto-created on init).

Token totals come from two places: BAML response usage (via `extract_baml_usage`) for LLM calls, and `EmbeddingResult.usage` / `RerankResult.usage` for consumer-supplied Protocol calls. The four-key vocabulary (`input` / `output` / `cache_creation` / `cache_read`) is shared with the rfnry agent SDK so a single admin UI consumes both.

Both fields default to a stderr-emitting sink. Pass `Observability(sink=NullSink())` / `Telemetry(sink=NullSink())` to silence; `None` is not accepted. Cross-cutting access is via `contextvars.ContextVar` (`current_obs()`, `current_query_row()`, `current_ingest_row()`) — set at the entry-point boundary, propagated automatically through async tasks. The library emits raw token counts only; no pricing tables, no cost calculator.

### LLM integration

All LLM calls go through BAML for structured output parsing, retry/fallback, and provider routing. Edit `baml/baml_src/`; regenerate with `poe baml:generate`. Never edit `baml_client/`.

`build_registry(ProviderClient)` (in `providers/registry.py`) builds a per-call BAML `ClientRegistry` with primary + optional fallback provider routing. The `ProviderClient` carries everything BAML needs: provider name, model, api_key, options dict (forwarded as-is), retry count, max_tokens, temperature, timeout_seconds.

### When to use BAML for a new feature

BAML's value is **structured-output parsing** with primary/fallback routing. The SDK keeps a small set of BAML functions, all substrate-only (vision extraction, entity extraction, index-time synthetic-query generation, situating-context generation, answer-quality judging, relevance gating, free-text generation).

Before adding a new BAML function, answer all 5. **Two or more "no" → don't use BAML.**

1. **Does the caller need a typed object, not a string?** If the consumer immediately stringifies the output or treats it as free text, BAML's structured-output value is wasted. *(`GenerateText` is the exception — it's the unstructured BAML wrapper consumers use to keep all LLM calls inside one orchestration layer.)*

2. **Is the schema a system boundary?** Does the parsed output flow into a store / index / mapper that requires specific shapes (`DiscoveredEntity`, `DetectedComponent`, `AnswerQualityJudgment`)?

3. **Will the caller get *more* useful as the model improves?** If better reasoning makes the call redundant, BAML wrapping is a deprecation magnet.

4. **Does the value justify the friction tax?** Adding a BAML function means: a `.baml` source file, a `poe baml:generate` regen step, a `baml_client/` diff, a `ClientRegistry` plumbing call, and contract tests touching it.

5. **Is this index-time augmentation, not query-time decision-making?** Index-time use compounds with model improvement. Query-time LLM-as-router/classifier/decomposer competes with the model.

## Key patterns

- **Protocol-based abstraction.** No inheritance; `Protocol` classes define interfaces (`BaseEmbeddings`, `BaseRetrievalMethod`, `BaseIngestionMethod`, `BaseReranking`, `BaseSparseEmbeddings`, `TokenCounter`). Any conforming object works.
- **No facades, no isinstance dispatch on providers.** The previous `Embeddings(ModelProvider)` / `Vision(ModelProvider)` / `Reranking(ModelProvider)` facades and the `AnthropicModelProvider` / `OpenAIModelProvider` / etc. union are gone. Consumer-supplied Protocol impls and `ProviderClient` cover the entire provider surface.
- **Modular pipeline.** Services receive `list[BaseRetrievalMethod]` / `list[BaseIngestionMethod]` and dispatch generically. Per-method error isolation.
- **Async-first.** All I/O is async. Services use `async def`; stores use asyncpg / aiosqlite.
- **Service pattern.** Each module has a `Service` class with dependencies injected via `__init__`.
- **Domain-neutral by default.** No hardcoded domain vocabulary in BAML prompts. Features needing vocabulary expose consumer-overridable config with empty defaults.

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
- Tests live in `tests/`, mirroring source layout.

## Config defaults and enforced bounds

`__post_init__` validators reject pathological values at construction time:

- `IngestionConfig.chunk_size_unit`: `Literal["chars", "tokens"]`, default `"tokens"`. With `chunk_size_unit="tokens"` and no `token_counter` supplied, `SemanticChunker` falls back to a whitespace word count and logs a warning.
- `IngestionConfig.token_counter`: `TokenCounter | None`, default `None` (consumer plugs in for accurate token counts).
- `IngestionConfig.parent_chunk_size`: sentinel `-1` (default) resolves to `3 * chunk_size`; explicit `0` disables parent-child indexing.
- `IngestionConfig.document_expansion`: nested `DocumentExpansionConfig`. Defaults disabled. When `enabled=True`, `provider_client` is required.
- `IngestionConfig.contextual_chunk`: nested `ContextualChunkConfig`. Defaults disabled. When `enabled=True`, both `provider_client` and `token_counter` are required.
- `RetrievalConfig.top_k`: `1 ≤ top_k ≤ 200`.
- `RoutingConfig.mode`: `QueryMode` enum, default `INDEXED`. Other values: `FULL_CONTEXT`, `AUTO`.
- `RoutingConfig.full_context_threshold`: `1_000 ≤ n ≤ 2_000_000`, default 150_000.
- `GenerationConfig`: `grounding_enabled=True` requires `grounding_threshold > 0` and a `provider_client`.
- `GenerationConfig.chunk_ordering`: `ChunkOrdering` enum, default `SCORE_DESCENDING`.
- `BenchmarkConfig.concurrency`: `1 ≤ n ≤ 20`, default 1.
- `DrawingIngestionConfig.dpi`: `150 ≤ dpi ≤ 600`, default 400.
- `DrawingIngestionConfig.relation_vocabulary`: every target must be in `ALLOWED_RELATION_TYPES`.
- `EntityIngestionConfig.entity_type_patterns`: regex strings compiled at `__post_init__`.
- `ProviderClient.timeout_seconds`: `> 0`, default 60.
- `ProviderClient.temperature`: `0.0 ≤ t ≤ 2.0`.
- `ProviderClient.max_retries`: `0 ≤ n ≤ 5`, default 3.
- `ProviderClient.context_size`: `int | None`, default `None`. When set, must be `≥ 1`. Used as a *safety cap*: `KnowledgeEngine.initialize()` refuses configs where `RoutingConfig.full_context_threshold + 16_000 (non-output reserve) + ProviderClient.max_tokens (output reserve)` exceeds it.
- `Neo4jGraphStore.password`: required.
- Public-input bounds: query ≤ 32 000 chars, `ingest_text` ≤ 5 000 000 chars, metadata ≤ 50 keys × 8 000 chars.
- `Source.ingestion_notes` (backed by `metadata["ingestion_notes"]`) records non-fatal pipeline events as `"<step>:<level>:<reason>"` strings. `Source.fully_ingested` is `True` iff that list is empty.

## Environment variables

The library reads only its own runtime toggles. No vendor API keys are read from the environment — the host application owns credential lookup and constructs `ProviderClient` accordingly.

- `KNWL_LOG_ENABLED=true` / `KNWL_LOG_LEVEL=DEBUG` — SDK logging.
- `KNWL_LOG_QUERIES=true` — include raw query text in logs (off by default; PII-safe). Use `rfnry_knowledge.common.logging.query_logging_enabled()` when adding new query-logging sites.
- `KNWL_BAML_LOG=info|warn|debug` — BAML runtime logging (SDK sets `BAML_LOG` from this).
- `BAML_LOG=info|warn|debug` — BAML runtime logging (direct override).
- `BOUNDARY_API_KEY` — Boundary collector key, process-global.
