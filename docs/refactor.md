# Refactor — slim, model-improvement-aligned redesign

A redesign of `rfnry-rag` around a single test: **does this feature compose with smarter models, or compete with them?** Anything that competes is on a deprecation clock and is removed. Anything that composes — pure infrastructure, protocol shape, structured input the model can exploit better — stays.

The result: ~9–10k LOC down from ~32k, one SDK instead of two, and a feature surface that gets *more* valuable as models get better instead of being undermined by them.

---

## Guiding principles

1. **Strip workarounds for old model limits.** Anything built to fake bigger context, smarter routing, or better instruction-following becomes dead weight as those limits disappear.
2. **Keep plumbing and protocols.** Async I/O, error isolation, pluggable methods, structured outputs (BAML), provider-agnostic facades — these are independent of model intelligence.
3. **Give the model better input, don't replace its judgment.** Drawings: extract structure once at ingest, emit cross-page references as graph data, let the model do the synthesis. Don't pre-compute the linkage ourselves.
4. **Compose with growing context windows.** `AUTO` routing transparently shifts to `FULL_CONTEXT` mode as corpora fit — the toolkit improves automatically as model context grows.
5. **One mental model.** Ingest → index → retrieve → ground → generate. No dual SDK, no service-class wrappers around single prompts.

---

## What we keep — feature, rationale, survival case

### Retrieval methods

| Feature | Why it survives next-gen models |
|---|---|
| **Vector retrieval** (dense + BM25 fused via internal RRF) | Embeddings are a *different* capability than generation — they index meaning at low cost so the model doesn't have to re-read the corpus every query. Embedding models improve independently of LLM size; BM25 is parameter-free math. Composes with the LLM, doesn't compete. |
| **Document retrieval** (Postgres FTS + substring) | Lexical matching is structurally different from semantic. Catches exact tokens (part numbers, error codes, identifiers) that embeddings smear. No LLM in the loop, ever. Orthogonal to model capability. |
| **Graph retrieval** (entity lookup + N-hop traversal) | Survives if and only if the graph holds real structural data (which our drawing ingestion produces). Better LLMs traverse better-quality graphs more powerfully — this scales up, not out. Improves with the model. |
| **RRF fusion + per-method weights** | Pure math. Doesn't know or care what model is downstream. Permanent. |
| **Pluggable method protocol** (`BaseRetrievalMethod`) | The only durable abstraction in retrieval: "here's a thing that returns ranked chunks." Future methods slot in without engine changes. Architectural, not algorithmic. |
| **Pipeline composition** (sequence + fuse retrieval methods) | Consumer-driven assembly. Whatever retrieval looks like in 2027, the user composes it from primitives. Plumbing, not policy. |

### Ingestion

| Feature | Why it survives |
|---|---|
| **Pluggable ingestion protocol** (`BaseIngestionMethod`) | Same logic as retrieval protocol. Architectural. |
| **Standard text/PDF/markdown ingestion** (chunking, parent-child, hashing, dedup) | Chunking is a function of *embedding* context (which has its own limits), not LLM context. Sane defaults + hash-based dedup are pure infrastructure. Independent of model size. |
| **Document expansion at index time** (synthetic queries → BM25 + embeddings) | Bridges user-vocabulary vs document-vocabulary at the *embedding* layer. The cost is paid once at ingest; the benefit is structural to lexical/dense retrieval, not to LLM intelligence. Improves with better embedding models. |
| **Drawing ingestion — per-page vision extraction** (LLM produces structured component lists, labels, off-page connector tags) | This is the part that *gets better as vision models get better*. We're not parsing pixels ourselves — we're asking the LLM to extract structure once at ingest, store it, and let downstream queries traverse it. Direct beneficiary of model improvement. |
| **Drawing ingestion — DXF native parse** (`ezdxf`, modelspace + paperspace per layout) | Reading native CAD is structurally better than vision-on-rasterized-CAD. No LLM involvement, no obsolescence path. Permanent. |
| **Drawing ingestion — structural cross-page graph emission** (every off-page connector tag, every label → graph edge candidate) | Replaces the old fuzzy/regex/LLM-residue linker. We *don't* link pages ourselves — we emit the linkage signals into the graph store and let the LLM resolve them at query time. The user-visible win ("ChatGPT loses cross-page context") becomes a model-strength story, not a heuristic-pipeline story. |
| **Vision-pipeline content cache** (file-hash + per-page-hash short-circuit) | Just sane caching — re-ingestion shouldn't re-pay for unchanged pages. No state machine, no resume protocol. Pure infrastructure. |

### Routing

| Feature | Why it survives |
|---|---|
| **`INDEXED` mode** (use the retrieval pipeline) | Default for corpora that exceed the model's window. Stays correct as long as some corpora are bigger than context. |
| **`FULL_CONTEXT` mode** (load whole corpus into prompt-cached prefix) | Gets *more* valuable as context windows grow and prompt caching matures. Composes with model improvements directly. |
| **`AUTO` routing** (corpus-token threshold dispatches between the two) | Deterministic threshold, not a guess. As windows grow, more corpora cross the threshold and transparently shift to `FULL_CONTEXT` without code changes. Self-improves with model generations. |

### Generation

| Feature | Why it survives |
|---|---|
| **Grounding gate** (refuse-when-context-irrelevant) | Even smart models hallucinate when given irrelevant context. The gate is a cheap pre-flight check, not a workaround. Stays useful at any model scale. |
| **Lost-in-the-middle chunk ordering** (score-descending / primacy-recency / sandwich) | ~50 LOC. U-shaped attention is empirical across all current frontier models; ordering is a free lever. If future models eliminate the U-shape, the default ordering still works. Cheap insurance. |

### Provider layer + structured I/O

| Feature | Why it survives |
|---|---|
| **Provider-agnostic facades** (`Embeddings`, `Vision`, `Reranking`, `LanguageModelClient`) | The *one* abstraction that genuinely matters in a multi-provider world: swap providers via config, never via code. Gets more valuable as the provider landscape fragments. |
| **BAML structured I/O** (typed inputs/outputs, retries, fallbacks) | Structured outputs don't go away — even with smarter models, schema-typed contracts are how production systems stay reliable. BAML also gives provider-agnostic prompt routing. Infrastructure that improves with the ecosystem. |
| **Prompt-injection fencing + contract test** | Injection vectors persist regardless of model intelligence (smarter models can be manipulated more cleverly, not less). The contract test is a CI guard, not feature code. Permanent. |

### Observability

| Feature | Why it survives |
|---|---|
| **Per-query trace** | The trace is the consumer's window into a non-deterministic pipeline. Better models don't make the pipeline observable. Permanent. |
| **Benchmark harness** | You can't claim retrieval works without a way to measure it. As models change, the harness lets consumers re-validate without reading code. Required regardless of model generation. |
| **Knowledge manager** (CRUD, scoping, corpus-token accounting) | Pure data management. The token-counting feeds `AUTO` routing. Infrastructure. |

### Cross-cutting

| Feature | Why it survives |
|---|---|
| **Async-first I/O** | Latency dominates. Async stays correct forever. |
| **Per-method error isolation** (one failing path doesn't break others) | Robustness pattern, not a feature. Permanent. |
| **Domain-neutral defaults + contract test** | No hardcoded vocabulary leaking into prompts. Lets the toolkit serve any domain without re-shipping. Permanent. |
| **CLI** (`rfnry-rag` mirrors SDK surface) | Consumer-facing operability. Doesn't depend on model capability. Permanent. |

---

## What's stripped — and why

| Removed | Reason |
|---|---|
| **`reasoning/` SDK entirely** (Analysis, Classification, Clustering, Compliance, Evaluation, Pipeline + their BAML + their CLI) | Service classes wrapping single prompts. Smarter models + reliable structured outputs make these recipes, not infrastructure. Anything worth keeping (multi-turn entity tracking, evaluation) folds into the retrieval SDK or docs. |
| **RAPTOR** (hierarchical summaries, blue/green tree rebuilds) | Existed because mid-2023 LLMs couldn't hold a corpus. With 1M-token context + prompt caching, the `FULL_CONTEXT` path dominates for the corpora where RAPTOR helped. |
| **Multi-hop iterative retrieval** | Hand-written decomposer + accumulator is a 2024 workaround. Frontier models with tool-use already do this internally. |
| **Tree search / TOC indexing** | Same as RAPTOR — exists to fake bigger context. |
| **HYBRID routing mode** | Overlapped with grounding gate + adaptive escalation. Two paths to "fall back to direct context" collapse to one. |
| **Adaptive top_k / LLM query classification** | Competes with the LLM's own judgment. Bigger model with more context doesn't need top_k tuned. |
| **Confidence-expansion retries** | Just set top_k higher; cost barely matters with caching. |
| **Failure-type classifier** (7 categories) | Better models fail less and more legibly. Trace shows what happened. |
| **Three query-rewriter variants** (HyDE / multi-query / step-back) | Most users pick one. Better models retrieve fine without rewriting. Keep one as default. |
| **Drawing ingestion's fuzzy/regex/LLM-residue cross-sheet linker** | Replaced by structural graph emission — let the model do cross-sheet reasoning at query time over assembled graph data. |
| **Reasoning-specific contract tests** | Gone with reasoning. |
| **`Pipeline` step composer (reasoning)** | It's `f(g(x))`. Strip. |

---

## Killing `common/`

The `common/` directory tells you *who uses* the code (everyone), not *what it is*. After the dual-SDK split is gone, there's no "common between SDKs" axis to organize around. Every file inside has a purpose-named home it can move to.

### `exceptions/` — one file per error family

`SdkBaseError` is dropped. With a single SDK, having both `SdkBaseError` and `RagError` is redundant — **`RagError` becomes the root** of the hierarchy.

```
src/rfnry_rag/exceptions/
├── __init__.py              # re-exports — `from rfnry_rag.exceptions import RagError`
├── base.py                  # RagError (root)
├── configuration.py         # ConfigurationError
├── ingestion.py             # IngestionError, ParseError, EmptyDocumentError, EmbeddingError, IngestionInterruptedError
├── retrieval.py             # RetrievalError
├── generation.py            # GenerationError
├── store.py                 # StoreError, DuplicateSourceError, SourceNotFoundError
└── input.py                 # InputError (RagError + ValueError)
```

### `providers/` — the LLM-and-friends provider layer

Anything provider-related (language, embeddings, vision, reranking) lives here. One concept, one folder.

```
src/rfnry_rag/providers/
├── __init__.py
├── client.py                # LanguageModelClient
├── provider.py              # LanguageModelProvider
├── registry.py              # build_registry (BAML ClientRegistry)
├── protocols.py             # BaseEmbeddings (only protocol that survives)
└── facades.py               # Embeddings, Vision, Reranking — runtime-dispatch facades
```

### `models/` — the domain types the SDK speaks

This is the **vocabulary of the SDK**: the plain dataclasses that get passed between layers. They're not algorithms, they're nouns. After the prune, the surviving types cluster cleanly:

- **Documents** — `Source` (an ingested file/text + its metadata), `Chunk` (a retrievable slice of a source).
- **Vectors** — `SparseVector`, `VectorPoint`, `VectorResult`. Vector-store DTOs that travel between the embeddings facade and the vector store.
- **Results** — `RetrievedChunk` (a chunk + score + provenance), `ContentMatch` (a document-store hit with FTS headline).
- **Stats** — `SourceStats` (counts and token totals used by the knowledge manager and `AUTO` routing).

Trace types (`RetrievalTrace`) move to `observability/trace.py` because they're an observability concern, not a domain noun.

```
src/rfnry_rag/models/
├── __init__.py              # re-exports
├── document.py              # Source, Chunk
├── vector.py                # SparseVector, VectorPoint, VectorResult
├── result.py                # RetrievedChunk, ContentMatch
└── stats.py                 # SourceStats
```

A folder (instead of a single `models.py`) makes each cluster discoverable and gives room for new domain types to land in the right neighbourhood instead of growing one mega-file.

### File-by-file relocation map

| Current location | What it is | New home |
|---|---|---|
| `common/errors.py` | `SdkBaseError`, `ConfigurationError` | `exceptions/base.py` (`RagError` replaces `SdkBaseError`), `exceptions/configuration.py` |
| `retrieval/common/errors.py` | `RagError` + 9 subclasses | Split across `exceptions/{ingestion,retrieval,generation,store,input}.py` |
| `common/language_model.py` | `LanguageModelClient`, `LanguageModelProvider`, `build_registry` | `providers/{client,provider,registry}.py` |
| `common/protocols.py` | `BaseEmbeddings`, `BaseSemanticIndex` | `BaseEmbeddings` → `providers/protocols.py`. `BaseSemanticIndex` → **delete** (only reasoning used it). |
| `common/embeddings.py` | `embed_batched` + batch-size constants | `ingestion/chunk/embeddings_batch.py` |
| `common/concurrency.py` | `run_concurrent` | `concurrency.py` at root |
| `common/logging.py` | `get_logger`, env-var propagation | `logging.py` at root |
| `common/startup.py` | BAML version check | `baml/version_check.py` |
| `common/cli.py` | `ConfigError`, `CONFIG_DIR`, `OutputMode`, `get_api_key`, `load_dotenv` | `cli/helpers.py` |
| `retrieval/common/models.py` | Domain dataclasses + `RetrievalTrace` | Domain types → `models/`. Trace → `observability/trace.py`. |
| `retrieval/common/hashing.py` | hash helpers | `ingestion/chunk/hashing.py` |
| `retrieval/common/page_range.py` | page-string parser | `ingestion/chunk/page_range.py` |
| `retrieval/common/formatting.py` | `chunks_to_context` | `generation/formatting.py` |
| `retrieval/common/grounding.py` | grounding helpers | `generation/grounding.py` |
| `retrieval/common/{concurrency,language_model,logging,startup}.py` | re-export shims | **delete** |
| `reasoning/common/*` | re-export shims | **delete with the reasoning SDK** |

## New folder structure

```
packages/python/
├── pyproject.toml
├── README.md
├── CLAUDE.md
├── CHANGELOG.md
├── uv.lock
├── src/rfnry_rag/
│   ├── __init__.py                    # top-level re-exports
│   ├── server.py                      # RagEngine — async context manager, dynamic pipeline assembly
│   ├── logging.py                     # get_logger, query_logging_enabled
│   ├── concurrency.py                 # run_concurrent
│   │
│   ├── exceptions/                    # one file per error family — `from rfnry_rag.exceptions import …`
│   │   ├── __init__.py                # re-exports
│   │   ├── base.py                    # RagError (root)
│   │   ├── configuration.py           # ConfigurationError
│   │   ├── ingestion.py               # IngestionError, ParseError, EmptyDocumentError, EmbeddingError, IngestionInterruptedError
│   │   ├── retrieval.py               # RetrievalError
│   │   ├── generation.py              # GenerationError
│   │   ├── store.py                   # StoreError, DuplicateSourceError, SourceNotFoundError
│   │   └── input.py                   # InputError (RagError + ValueError)
│   │
│   ├── providers/                     # LLM + embeddings + vision + reranking provider layer
│   │   ├── __init__.py
│   │   ├── client.py                  # LanguageModelClient
│   │   ├── provider.py                # LanguageModelProvider
│   │   ├── registry.py                # build_registry (BAML ClientRegistry)
│   │   ├── protocols.py               # BaseEmbeddings
│   │   └── facades.py                 # Embeddings, Vision, Reranking — runtime-dispatch facades
│   │
│   ├── models/                        # domain dataclasses — the vocabulary the SDK speaks
│   │   ├── __init__.py                # re-exports
│   │   ├── document.py                # Source, Chunk
│   │   ├── vector.py                  # SparseVector, VectorPoint, VectorResult
│   │   ├── result.py                  # RetrievedChunk, ContentMatch
│   │   └── stats.py                   # SourceStats
│   │
│   ├── config/                        # all config dataclasses, one place
│   │   ├── __init__.py
│   │   ├── server.py                  # RagServerConfig
│   │   ├── persistence.py             # PersistenceConfig
│   │   ├── ingestion.py               # IngestionConfig, DrawingIngestionConfig, GraphIngestionConfig, DocumentExpansionConfig
│   │   ├── retrieval.py               # RetrievalConfig
│   │   ├── routing.py                 # RoutingConfig (INDEXED / FULL_CONTEXT / AUTO)
│   │   ├── generation.py              # GenerationConfig (grounding, chunk_ordering)
│   │   └── benchmark.py               # BenchmarkConfig
│   │
│   ├── ingestion/                     # ingestion pipeline
│   │   ├── __init__.py
│   │   ├── base.py                    # BaseIngestionMethod protocol
│   │   ├── service.py                 # IngestionService — generic dispatch
│   │   ├── chunk/                     # chunking + per-chunk concerns
│   │   │   ├── __init__.py
│   │   │   ├── chunker.py
│   │   │   ├── parent_child.py
│   │   │   ├── hashing.py
│   │   │   ├── page_range.py
│   │   │   ├── embeddings_batch.py    # embed_batched (was common/embeddings.py)
│   │   │   └── expansion.py           # synthetic-query document expansion
│   │   ├── methods/
│   │   │   ├── __init__.py
│   │   │   ├── vector.py              # VectorIngestion
│   │   │   ├── document.py            # DocumentIngestion
│   │   │   └── graph.py               # GraphIngestion (consumer-agnostic mapper)
│   │   └── drawing/                   # drawing ingestion (the differentiated path)
│   │       ├── __init__.py
│   │       ├── service.py             # DrawingIngestionService — render → extract → ingest
│   │       ├── render.py              # PDF page render, DXF layout enumeration
│   │       ├── extract_pdf.py         # vision LLM per-page structured extraction
│   │       ├── extract_dxf.py         # ezdxf native parse, modelspace + paperspace
│   │       └── graph_emit.py          # emit cross-page connector tags + labels as graph edges
│   │
│   ├── retrieval/                     # retrieval pipeline
│   │   ├── __init__.py
│   │   ├── base.py                    # BaseRetrievalMethod protocol
│   │   ├── service.py                 # RetrievalService — runs methods concurrently, fuses
│   │   ├── fusion.py                  # RRF + per-method weights
│   │   ├── methods/
│   │   │   ├── __init__.py
│   │   │   ├── vector.py              # VectorRetrieval (dense + BM25 fused internally)
│   │   │   ├── document.py            # DocumentRetrieval (FTS + substring)
│   │   │   └── graph.py               # GraphRetrieval (entity lookup + N-hop)
│   │   ├── reranking.py               # cross-encoder reranking (Cohere, Voyage)
│   │   └── routing.py                 # AUTO threshold dispatch (INDEXED vs FULL_CONTEXT)
│   │
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── service.py                 # GenerationService
│   │   ├── grounding.py               # grounding gate
│   │   ├── formatting.py              # chunks_to_context
│   │   ├── ordering.py                # chunk_ordering (score-desc / primacy-recency / sandwich)
│   │   └── full_context.py            # FULL_CONTEXT path — corpus-into-prompt with caching
│   │
│   ├── stores/
│   │   ├── __init__.py
│   │   ├── vector/
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   └── qdrant.py              # QdrantVectorStore
│   │   ├── document/
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   ├── postgres.py            # PostgresDocumentStore (FTS, substring)
│   │   │   └── filesystem.py
│   │   ├── graph/
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   ├── neo4j.py               # Neo4jGraphStore
│   │   │   └── mapper.py              # GraphIngestion config (consumer-agnostic patterns)
│   │   └── metadata/
│   │       ├── __init__.py
│   │       └── sqlalchemy.py          # SQLAlchemyMetadataStore
│   │
│   ├── knowledge/
│   │   ├── __init__.py
│   │   ├── manager.py                 # KnowledgeManager — CRUD + corpus-token accounting
│   │   └── migration.py
│   │
│   ├── observability/
│   │   ├── __init__.py
│   │   ├── trace.py                   # RetrievalTrace
│   │   ├── benchmark.py               # BenchmarkReport, run_cases
│   │   └── metrics.py                 # ExactMatch, F1Score, RetrievalRecall, RetrievalPrecision, LLMJudgment
│   │
│   ├── cli/
│   │   ├── __init__.py
│   │   ├── helpers.py                 # ConfigError, CONFIG_DIR, OutputMode, get_api_key, load_dotenv
│   │   ├── commands/
│   │   │   ├── __init__.py
│   │   │   ├── ingest.py
│   │   │   ├── query.py
│   │   │   ├── benchmark.py
│   │   │   └── knowledge.py
│   │   ├── config.py                  # config loader (~/.config/rfnry_rag/)
│   │   └── output.py                  # formatters
│   │
│   └── baml/                          # one BAML tree, no SDK split
│       ├── version_check.py           # BAML version-mismatch guard (was common/startup.py)
│       ├── baml_src/
│       │   ├── clients.baml
│       │   ├── grounding.baml
│       │   ├── expansion.baml         # document expansion synthetic queries
│       │   ├── extract_drawing.baml   # vision drawing extraction
│       │   ├── judge.baml             # LLMJudgment metric
│       │   └── rewrite.baml           # single query rewriter (multi-query)
│       └── baml_client/               # generated
│
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── ingestion/
    │   ├── test_chunk.py
    │   ├── test_methods.py
    │   ├── test_drawing.py
    │   └── test_expansion.py
    ├── retrieval/
    │   ├── conftest.py
    │   ├── test_methods.py
    │   ├── test_fusion.py
    │   ├── test_routing.py
    │   └── test_reranking.py
    ├── generation/
    │   ├── test_grounding.py
    │   ├── test_ordering.py
    │   └── test_full_context.py
    ├── stores/
    ├── observability/
    │   ├── test_trace.py
    │   └── test_benchmark.py
    └── contracts/
        ├── test_baml_prompt_fence_contract.py
        ├── test_baml_prompt_domain_agnostic.py
        └── test_config_bounds_contract.py
```

### Structural notes

- **No `common/` directory anywhere.** Every module name describes a concept (`exceptions`, `providers`, `models`, `observability`) rather than an audience.
- **No more dual-SDK split.** `retrieval/`, `ingestion/`, `generation/`, `stores/`, `observability/` are siblings under `rfnry_rag/`. The `reasoning/` directory is gone entirely.
- **`config/` is centralized.** One import path (`from rfnry_rag.config import RagServerConfig, RoutingConfig, ...`) and one place for the bounds-contract test to walk.
- **`drawing/` lives under `ingestion/`** as a peer to `methods/`, not a separate concept. It produces the same chunk/vector/graph outputs the standard methods do. Its differentiator (cross-page graph emission) is a single file.
- **`observability/` is its own top-level concern.** Trace, benchmark, and metrics belong together.
- **`baml/` is one tree.** No retrieval/reasoning split; one set of BAML clients regenerated by one `poe` task. The startup version-check moves here too.
- **`server.py` stays at the top of the SDK.** `RagEngine` is the sole entry point; it composes `IngestionService`, `RetrievalService`, `GenerationService` per config.

---

## Estimated impact

- **~9,000–10,000 LOC** of source (down from ~32,000).
- **~12 config dataclasses** (down from ~25).
- **~4 contract tests** retained (cheap CI insurance).
- **One mental model**: ingest → index → retrieve → ground → generate, with `AUTO` routing skipping retrieval when the corpus fits.

---

## The thesis, restated

Every component on the keep-list either (a) is pure infrastructure that the model never touches, (b) provides structured input that better models exploit better, or (c) gets out of the way (`AUTO` routing, `FULL_CONTEXT` mode) when the model is capable enough not to need us. Nothing on this list competes with the model's growing capability.

That's the test that lets this codebase stay relevant through Sonnet 5, GPT-6, Gemini 3, and whatever follows.
