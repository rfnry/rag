# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While the project is at 0.x, breaking changes may land in any release. Pin
exact versions in production until 1.0.

## [0.1.0] — 2026-05-01

Initial public foundation. Modular RAG toolkit for Python:

- **Modular ingestion + retrieval.** Vector (dense + BM25 fused internally), document (Postgres FTS + substring), and graph (entity lookup + N-hop traversal). Each method implements `BaseIngestionMethod` / `BaseRetrievalMethod`; methods compose generically through `IngestionService` / `RetrievalService` with reciprocal-rank fusion and per-method error isolation.
- **Routing layer.** `RoutingConfig.mode` selects between `INDEXED`, `FULL_CONTEXT` (whole corpus into a prompt-cached prefix, no retrieval), and `AUTO` (corpus-token threshold dispatches per query). `LanguageModel.context_size` acts as a safety cap: engine init refuses configurations where `full_context_threshold + reserve` would overflow the declared window.
- **Index-time enrichments.** Opt-in `chunk_context_headers` (templated structural prefix), `DocumentExpansionConfig` (LLM-generated synthetic queries per chunk), and `ContextualChunkConfig` (Anthropic Contextual Retrieval — situating-context blob per chunk, prompt-cache-aware Anthropic backend).
- **Drawing pipeline.** Vision LLM extracts components, labels, and off-page connector tags from schematic PDFs; cross-page references emit into the graph store as edge candidates. DXF files parse natively through `ezdxf`. Symbol vocabularies (IEC 60617 + ISA 5.1) ship as overridable defaults. Per-page vision failures soft-skip the page rather than aborting the source.
- **Generation.** Grounding gate refuses to answer on irrelevant context. Chunk ordering supports score-descending (default), primacy-recency, and sandwich layouts to mitigate lost-in-the-middle. Native SDK dispatch for plain text generation; BAML for structured outputs.
- **Observability.** `Source.ingestion_notes` records non-fatal degradations during ingest; `Source.fully_ingested` is the boolean derived view. `EnrichmentSkipped` typed exception signals "optional enrichment couldn't run for this source". `KnowledgeManager.health(source_id)` fuses ingestion notes, retrieval stats (hits / grounded / ungrounded), and embedding-model freshness. `RetrievalTrace` captures per-stage state on opt-in. Benchmark harness aggregates exact match, F1, retrieval recall/precision, and optional LLM-judge scores with per-case traces.
- **Providers.** Anthropic, OpenAI, Gemini for generation and vision. OpenAI, Voyage, Cohere for embeddings. Cohere, Voyage for cross-encoder reranking. Provider identity sits on `LanguageModel`; `LanguageModelClient` adds retry/fallback/timeout for BAML-routed structured calls.
- **Hardening.** Prompt-injection fence contract test runs on every BAML source. Domain-agnostic prompt contract test rejects banned vocabulary. Numeric config fields are bounded with explicit ranges or marked `# unbounded: <reason>`; a contract test enforces this. Public-input length caps on queries, ingestion text, and metadata.
- **Tooling.** CLI subcommands `ingest`, `query`, `benchmark`, `knowledge` (list / inspect / remove). TOML config loader at `~/.config/rfnry_rag/config.toml`. Async-first; all I/O uses asyncpg / aiosqlite / async SDK clients.
