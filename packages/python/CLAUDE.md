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
│   ├── errors.py        # BaseException, ConfigurationError (base classes)
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
   - **GraphRetrieval** — Entity lookup + N-hop traversal (requires graph store)
   - **Enrich** — Structured retrieval with field filtering (requires metadata store)
   - **Tree** — LLM reasoning over hierarchical document structure (requires metadata store + `TreeSearchConfig.enabled`)
3. **Reranking** (optional) — Cross-encoder reranking against original query (Cohere, Voyage)
4. **Chunk refinement** (optional) — Extractive (context window) or abstractive (LLM summarization) refinement
5. **Generation** (for `query()` only) — Grounding gate → LLM relevance gate → optional clarification → LLM generation

### Modular Pipeline

Retrieval and ingestion are protocol-based plugin architectures. No mandatory vector DB or embeddings — at least one retrieval path (vector, document, or graph) must be configured.

- **`BaseRetrievalMethod`** / **`BaseIngestionMethod`** — Protocol interfaces in `modules/retrieval/base.py` and `modules/ingestion/base.py`. Any conforming class works.
- **Method classes** — `VectorRetrieval`, `DocumentRetrieval`, `GraphRetrieval` (retrieval); `VectorIngestion`, `DocumentIngestion`, `GraphIngestion`, `TreeIngestion` (ingestion). Each is self-contained with error isolation and timing logs.
- **`MethodNamespace[T]`** — Generic container exposing methods as attributes (`rag.retrieval.vector`) and supporting iteration (`for m in rag.retrieval`).
- **Dynamic assembly** — `RagEngine.initialize()` builds method lists from config, validates cross-config constraints via `_validate_config()`, assembles `RetrievalService` and `IngestionService` with method list dispatch.
- **`AnalyzedIngestionService`** — 3-phase LLM pipeline (analyze → synthesize → ingest) for vision-analyzed documents. Uses `graph_store` directly for pre-extracted entities, delegates document storage to method list.

### Error Hierarchy

```
BaseException (common base)
├── ConfigurationError (shared)
├── RagError (retrieval)
│   ├── IngestionError, ParseError, EmptyDocumentError, EmbeddingError, IngestionInterruptedError, TreeIndexingError
│   ├── RetrievalError, GenerationError, TreeSearchError
│   └── StoreError, DuplicateSourceError, SourceNotFoundError
└── ReasoningError (reasoning)
    ├── AnalysisError, ClassificationError, ClusteringError
    ├── ComplianceError, EvaluationError
```

### LLM Integration

All LLM calls go through BAML for structured output parsing, retry/fallback policies, and observability. Each SDK has its own `baml_src/` (source definitions) and `baml_client/` (auto-generated — never edit). After modifying `.baml` files, regenerate with `poe baml:generate:retrieval` or `poe baml:generate:reasoning`.

`LanguageModelClient` in `common/language_model.py` builds a BAML `ClientRegistry` with primary + optional fallback provider routing. `LanguageModelProvider` configures a single provider endpoint (API key, base URL, model).

## Key Patterns

- **Protocol-based abstraction** — No inheritance; `Protocol` classes define interfaces (`BaseEmbeddings` (in `rfnry_rag.common.protocols`), `BaseSemanticIndex`, `BaseReranking`, `BaseRetrievalMethod`, `BaseIngestionMethod`, `BaseQueryRewriting`, `BaseChunkRefinement`, `BaseRetrievalJudgment`, etc.). Any conforming object works.
- **Facade pattern** — `Embeddings(LanguageModelProvider)`, `Vision(LanguageModelProvider)`, and `Reranking(LanguageModelProvider | LanguageModelClient)` are public facades that select the correct private provider implementation at runtime. Concrete providers (e.g. OpenAI, Voyage, Cohere) are private.
- **Modular pipeline** — Retrieval and ingestion methods are pluggable. Services receive `list[BaseRetrievalMethod]` / `list[BaseIngestionMethod]` and dispatch generically. Methods carry `weight` and `top_k` configuration. Per-method error isolation (catch, log, continue).
- **Async-first** — All I/O is async. Services use `async def`, stores use asyncpg/aiosqlite.
- **Service pattern** — Each module has a `Service` class with dependencies injected via `__init__`.
- **Shared common, SDK-specific re-exports** — SDK `common/` modules are thin re-exports from `rfnry_rag.common`. Retrieval-specific utilities (models, formatting, hashing, page_range) stay in retrieval's own `common/`.
- **Config dataclasses** — Pydantic V2 or plain dataclasses with `__post_init__` validation. `PersistenceConfig.vector_store` and `IngestionConfig.embeddings` are optional — at least one retrieval path must be configured.

## Linting & Style

- Ruff: line-length 120, target py312, rules: E, F, I, UP, B, SIM
- MyPy: python 3.12, ignores missing imports
- Both tools exclude `baml_client/` directories

## Testing

- pytest with `asyncio_mode = "auto"` — no `@pytest.mark.asyncio` needed
- Tests use `AsyncMock` and `SimpleNamespace` for lightweight mocking
- Tests in `tests/` subdirectories within each SDK + inline `test_*.py` in some modules
- 548 tests total across both SDKs

## Environment Variables

- `RFNRY_RAG_LOG_ENABLED=true` / `RFNRY_RAG_LOG_LEVEL=DEBUG` — SDK logging
- `RFNRY_RAG_BAML_LOG=info|warn|debug` — BAML runtime logging (SDK sets `BAML_LOG` from this)
- `BAML_LOG=info|warn|debug` — BAML runtime logging (direct override)
- Config lives at `~/.config/rfnry_rag/config.toml` + `.env`
