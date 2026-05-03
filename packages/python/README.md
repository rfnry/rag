# rfnry-knowledge

A modular, **provider-agnostic** retrieval engine for Python. Compose vector, document, and graph retrieval methods into one pipeline, fuse their results, and route between indexed retrieval and full-context generation based on corpus size — automatically. The engine ships zero provider implementations: the host application brings any LLM, embedder, or reranker that conforms to the library's Protocols and plugs it in. Built around a single principle: as language models grow stronger and contexts grow longer, the toolkit gets out of their way instead of working around them.

> **v0.1.0** — early foundation. Breaking changes are possible in any 0.x release. Pin exact versions in production until 1.0.

## Get Started

Install with [uv](https://docs.astral.sh/uv/):

```bash
uv add rfnry-knowledge                  # core SDK
uv add "rfnry-knowledge[graph]"         # + Neo4j graph support
```

The library is SDK-only. There is no CLI; the host application owns all transport (HTTP, CLI, queue worker, etc.).

A minimal vector + document retrieval pipeline. Note that `Embeddings` and `ProviderClient` LLM construction is the consumer's concern — the snippet below assumes a sibling `my_providers` module that returns objects matching `BaseEmbeddings` and `ProviderClient`:

The engine is organized around three peer retrieval pillars — **Semantic**, **Keyword**, **Entity** — that run in parallel inside `QueryMode.RETRIEVAL` and merge via reciprocal rank fusion. `QueryMode.DIRECT` skips retrieval entirely and loads the full corpus into a prompt-cached prefix; `QueryMode.AUTO` picks per query based on a corpus-token threshold.

```python
import asyncio

from pydantic import SecretStr

from rfnry_knowledge import (
    EntityIngestion,
    EntityRetrieval,
    GenerationConfig,
    IngestionConfig,
    KeywordIngestion,
    KeywordRetrieval,
    KnowledgeEngine,
    KnowledgeEngineConfig,
    Neo4jGraphStore,
    PostgresDocumentStore,
    ProviderClient,
    QdrantVectorStore,
    QueryMode,
    RetrievalConfig,
    RoutingConfig,
    SemanticIngestion,
    SemanticRetrieval,
    SQLAlchemyMetadataStore,
)

# Consumer-supplied; library defines BaseEmbeddings as a Protocol only.
from my_providers import build_embeddings


async def main() -> None:
    embeddings = build_embeddings()  # any BaseEmbeddings-conforming object

    generation_client = ProviderClient(
        name="anthropic",                       # BAML-recognized provider key
        model="claude-sonnet-4-5",
        api_key=SecretStr("sk-ant-…"),
        max_tokens=4096,
        temperature=0.0,
        context_size=200_000,
    )

    vector_store = QdrantVectorStore(url="http://localhost:6333", collection="docs")
    document_store = PostgresDocumentStore(url="postgresql+asyncpg://…")
    graph_store = Neo4jGraphStore(uri="bolt://localhost:7687", username="neo4j", password="…")
    metadata_store = SQLAlchemyMetadataStore(url="postgresql+asyncpg://…")

    config = KnowledgeEngineConfig(
        metadata_store=metadata_store,
        ingestion=IngestionConfig(
            methods=[
                SemanticIngestion(store=vector_store, embeddings=embeddings),
                KeywordIngestion(store=document_store),
                EntityIngestion(store=graph_store, provider_client=generation_client),
            ],
        ),
        retrieval=RetrievalConfig(
            methods=[
                # Semantic pillar — dense + optional sparse hybrid
                SemanticRetrieval(store=vector_store, embeddings=embeddings),
                # Keyword pillar — pick a backend per call site (BM25 or Postgres FTS)
                KeywordRetrieval(backend="postgres_fts", document_store=document_store),
                # Entity pillar — entity lookup + N-hop traversal over the graph store
                EntityRetrieval(store=graph_store),
            ],
            top_k=8,
        ),
        generation=GenerationConfig(provider_client=generation_client, grounding_enabled=True),
        routing=RoutingConfig(mode=QueryMode.AUTO, full_context_threshold=150_000),
    )

    async with KnowledgeEngine(config) as engine:
        await engine.ingest("manual.pdf", knowledge_id="equipment")
        result = await engine.query("how do I replace the filter?", knowledge_id="equipment")
        print(result.answer)


asyncio.run(main())
```

A complete factory-operations example with vector + document + graph + drawing ingestion lives at [`yard/examples/rfnry-knowledge/operation-assistant/`](../../yard/examples/rfnry-knowledge/operation-assistant/).

---

## Provider contract

The library defines four Protocols and one `ProviderClient` dataclass. The consumer implements/instantiates each.

| Surface | Contract | Consumer responsibility |
|---|---|---|
| LLM generation, structured outputs (BAML), grounding gate, vision (BAML-routed) | `ProviderClient(name, model, api_key, options, max_retries, max_tokens, temperature, timeout_seconds, context_size, fallback, strategy)` | Pick a BAML-recognized `name` (`"anthropic"` / `"openai"` / `"google"` / etc.) and pass credentials. The engine never imports vendor SDKs directly. |
| Embeddings | `BaseEmbeddings` Protocol — `name`, `model`, `embed(texts) -> EmbeddingResult`, `embedding_dimension()` | Wrap any embedding API or local model. Return `EmbeddingResult(vectors, usage=TokenUsage(...))`; the engine accumulates `usage` into telemetry rows. |
| Sparse embeddings | `BaseSparseEmbeddings` Protocol — `embed_sparse`, `embed_sparse_query` | Wrap any sparse embedder (FastEmbed, Splade, etc.). |
| Reranking | `BaseReranking` Protocol — `rerank(query, results, top_k) -> RerankResult` | Wrap any cross-encoder rerank API. |
| Token counting | `TokenCounter` Protocol — `count(text) -> int` | Wrap tiktoken or any tokenizer. Without one, token-mode chunking falls back to whitespace word count. |

`TokenUsage` is a TypedDict with four keys: `input`, `output`, `cache_creation`, `cache_read`. Missing keys default to zero. The same shape mirrors the `rfnry` agent SDK so a single admin UI consumes telemetry from both.

---

## Features

### Retrieval

**Three peer pillars composed in parallel.** `SemanticRetrieval` (dense + optional sparse hybrid), `KeywordRetrieval` (lexical match — `backend="bm25"` runs in-memory over the vector store, `backend="postgres_fts"` against the document store), and `EntityRetrieval` (entity lookup + N-hop traversal over the graph store). All three implement `BaseRetrievalMethod` and run concurrently per query, merging through reciprocal rank fusion with per-method weights. No mandatory pillar; configure only the paths you need. Per-method error isolation means one failing path does not break the others.

**Auto routing between retrieval and full-context generation.** Each query is dispatched through one of three modes: `RETRIEVAL` (run the three pillars in parallel and fuse), `DIRECT` (load the entire corpus into a prompt-cached prefix and let the model answer directly), or `AUTO` (a corpus-token threshold dispatches between them per query). When `ProviderClient.context_size` is declared, engine init refuses configurations where the threshold plus reserve would overflow the model's window — the cap is a safety bound, not a routing target.

**Cross-encoder reranking.** Optional reranking against the original query, via any `BaseReranking`-conforming object the consumer supplies. Sits cleanly between fusion and generation; opt-in per config.

### Ingestion

**Pillar-mirrored ingestion methods.** `SemanticIngestion` (writes embeddings to the vector store), `KeywordIngestion` (writes raw content to the document store, backing the Postgres-FTS keyword backend), and `EntityIngestion` (writes entities + relations to the graph store). All implement `BaseIngestionMethod`. Required vs optional methods are part of the contract: required-method failures abort the ingest; optional methods log and continue. The BM25 keyword backend reads vector-store payloads directly at query time and needs no separate ingestion.

**Drawing-aware ingestion for diagram-first documents.** Schematics, P&ID, wiring, and mechanical drawings break in chunk-and-pray pipelines — page 2 loses its connection to page 5. The drawing pipeline takes a different path: a vision call (BAML `AnalyzeDrawingPage`, routed via the consumer's `ProviderClient`) extracts structure once per page (components, labels, off-page connector tags) and emits every cross-page reference into the graph store as an edge candidate. Cross-sheet reasoning happens at query time, *over the assembled graph*, by the model itself. DXF files parse natively through `ezdxf` with no LLM calls. Symbol vocabularies (IEC 60617 + ISA 5.1 ship as defaults) are consumer-overridable. Per-page vision failures soft-skip the page and record a note rather than aborting the whole ingest.

**Index-time enrichment.** Three orthogonal opt-ins:
- `chunk_context_headers` — templated `Document: X | Type: Y | Page: N | Section: …` prepended to chunk text.
- `DocumentExpansionConfig` — LLM generates synthetic queries each chunk could answer (docT5query-style); flows into both BM25 and embeddings. Routes through BAML.
- `ContextualChunkConfig` — Anthropic's Contextual Retrieval recipe; LLM generates a 50–100 token blob situating each chunk within its source document, prepended before embedding/BM25. Routes through BAML.

### Generation

**Grounding gate.** Before the final LLM call, a relevance gate scores retrieved context against the query and refuses to answer when context is irrelevant rather than hallucinating from low-confidence chunks.

**Lost-in-the-middle mitigation.** Generation context can be assembled in score-descending order (default), primacy-recency, or sandwich. The non-default orderings put high-confidence chunks where U-shaped attention actually uses them.

**Long-context direct generation.** When the corpus fits the model's window, `QueryMode.DIRECT` loads the full corpus into a stable prompt prefix optimized for prompt-cache hits. Pairs cleanly with `AUTO` routing for transparent retrieval-or-direct dispatch.

### Observability

**Per-source health view.** `Source.fully_ingested` plus `Source.ingestion_notes: list[str]` (entries formatted `<step>:<level>:<reason>`) record any non-fatal degradation that happened during ingest — a contextualization skip on an oversized document, a vision page failure, a graph-extraction partial. `KnowledgeManager.health(source_id)` fuses these with `SourceStats` (retrieval hits, grounded vs ungrounded answer counts) and the `stale` flag (embedding-model migration).

**Per-query trace.** Pass `trace=True` to receive a `RetrievalTrace` capturing rewritten queries, per-method results (keyed by method name, including empty-result methods), fusion output, reranking, grounding decision, routing decision, and per-stage timings. Default `trace=False` is byte-for-byte unchanged.

**Benchmark harness.** Structured test cases run through `KnowledgeEngine.benchmark()`. Aggregates exact match, F1, retrieval recall and precision (when expected source IDs are provided), and optional LLM-judge scores. Per-case traces are part of the report so individual failures are debuggable.

**Structured event stream.** `Observability` on `KnowledgeEngineConfig` is always-on. Every entry point, every LLM call, and every retrieval / ingestion method emit a typed `ObservabilityRecord` (Pydantic, JSON-serializable) through the configured `Sink`. Default `JsonlStderrSink` writes one JSON line per record; swap to `JsonlFileSink`, `MultiSink`, or a custom `Sink` without touching pipeline code. `kind` discriminates events: `query.start`, `provider.call`, `retrieval.method.success`, etc. Pass `Observability(sink=NullSink())` to silence.

**Row-per-transaction telemetry.** `Telemetry` writes one `QueryTelemetryRow` per query and one `IngestTelemetryRow` per ingest. Rows carry `outcome`, `duration_ms`, per-method timings, raw token counts (input / output / cache_creation / cache_read), and routing/grounding decisions. Default sink is stderr; use `SqlAlchemyTelemetrySink(metadata_store)` to persist into `knowledge_query_telemetry` / `knowledge_ingest_telemetry` tables for admin-UI consumption. Token usage is fed by the consumer's `EmbeddingResult` / `RerankResult.usage` and by BAML's response usage metadata; the library emits raw counts only and never computes cost.

### Provider integration

**No vendor coupling.** The library imports zero vendor SDKs. There is no `anthropic`, `openai`, `google-genai`, `voyageai`, `cohere`, `tiktoken`, or `fastembed` dependency. The `BaseEmbeddings` / `BaseSparseEmbeddings` / `BaseReranking` / `TokenCounter` Protocols + `ProviderClient` dataclass form the entire provider surface; the host application brings the runtime.

**BAML-routed structured outputs.** Vision page analysis, entity extraction, document synthesis, situating-context generation, the answer-quality judge, and free-text generation all go through BAML. The `ProviderClient` populates a per-call BAML `ClientRegistry` (via `build_registry(client)`); BAML handles primary/fallback routing, retry policies, and provider-specific transport. Adding a provider means adding a `ProviderClient(name="<baml-provider>", …)` — no library change.

**Prompt-injection-resistant fencing.** Every user-controlled prompt parameter is wrapped with explicit start/end markers and a "treat as untrusted" directive. A contract test scans every BAML source file and fails CI if any function ships a user-input parameter unfenced. A second contract test scans for banned domain vocabulary and fails CI if any leak into prompts.

---

## License

MIT — see [LICENSE](../../LICENSE).
