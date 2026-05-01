# rfnry-rag

A modular retrieval toolkit for Python. Compose vector, document, and graph retrieval methods into one pipeline, fuse their results, and route between indexed retrieval and full-context generation based on corpus size — automatically. Built around a single principle: as language models grow stronger and contexts grow longer, the toolkit gets out of their way instead of working around them.

> **v0.1.0** — early foundation. Breaking changes are possible in any 0.x release. Pin exact versions in production until 1.0.

## Get Started

Install with [uv](https://docs.astral.sh/uv/):

```bash
uv add rfnry-rag                  # core SDK
uv add "rfnry-rag[graph]"         # + Neo4j graph support
uv add "rfnry-rag[cli]"           # + command-line interface
```

A minimal vector + document retrieval pipeline:

```python
import asyncio
import os

from rfnry_rag import (
    DocumentIngestion,
    DocumentRetrieval,
    Embeddings,
    GenerationConfig,
    IngestionConfig,
    LanguageModel,
    LanguageModelClient,
    PostgresDocumentStore,
    QdrantVectorStore,
    QueryMode,
    RagEngine,
    RagEngineConfig,
    RetrievalConfig,
    RoutingConfig,
    SQLAlchemyMetadataStore,
    VectorIngestion,
    VectorRetrieval,
)


async def main() -> None:
    embeddings = Embeddings(LanguageModel(
        provider="openai",
        model="text-embedding-3-small",
        api_key=os.environ["OPENAI_API_KEY"],
    ))
    generation = LanguageModelClient(lm=LanguageModel(
        provider="anthropic",
        model="claude-sonnet-4-5",
        api_key=os.environ["ANTHROPIC_API_KEY"],
        context_size=200_000,
    ))

    vector_store = QdrantVectorStore(url="http://localhost:6333", collection="docs")
    document_store = PostgresDocumentStore(url=os.environ["POSTGRES_URL"])
    metadata_store = SQLAlchemyMetadataStore(url=os.environ["POSTGRES_URL"])

    config = RagEngineConfig(
        metadata_store=metadata_store,
        ingestion=IngestionConfig(
            methods=[
                VectorIngestion(store=vector_store, embeddings=embeddings),
                DocumentIngestion(store=document_store),
            ],
            embeddings=embeddings,
        ),
        retrieval=RetrievalConfig(
            methods=[
                VectorRetrieval(store=vector_store, embeddings=embeddings, bm25_enabled=True),
                DocumentRetrieval(store=document_store),
            ],
            top_k=8,
        ),
        generation=GenerationConfig(lm_client=generation, grounding_enabled=True),
        routing=RoutingConfig(mode=QueryMode.AUTO, full_context_threshold=150_000),
    )

    async with RagEngine(config) as rag:
        await rag.ingest("manual.pdf", knowledge_id="equipment")
        result = await rag.query("how do I replace the filter?", knowledge_id="equipment")
        print(result.answer)


asyncio.run(main())
```

CLI mirrors the SDK surface (`rfnry-rag ingest …` / `rfnry-rag query …` / `rfnry-rag benchmark …` / `rfnry-rag knowledge inspect …`).

A complete factory-operations example with vector + document + graph + drawing ingestion lives at [`yard/examples/rfnry-rag/operation-assistant/`](../../yard/examples/rfnry-rag/operation-assistant/).

---

## Features

### Retrieval

**Modular method composition.** Vector (dense + BM25 fused internally), document (Postgres FTS + substring), and graph (entity lookup + N-hop traversal). All paths are pluggable via `BaseRetrievalMethod` and run concurrently per query, merging through reciprocal rank fusion with per-method weights. No mandatory backend; configure only the paths you need. Per-method error isolation means one failing path does not break the others.

**Auto routing between indexed retrieval and full-context generation.** Each query is dispatched through one of three modes: `INDEXED` (the standard retrieval pipeline), `FULL_CONTEXT` (load the entire corpus into a prompt-cached prefix and let the model answer directly), or `AUTO` (a corpus-token threshold dispatches between them per query). When a generation `LanguageModel.context_size` is declared, engine init refuses configurations where the threshold plus reserve would overflow the model's window — the cap is a safety bound, not a routing target.

**Cross-encoder reranking.** Optional reranking against the original query (Cohere, Voyage). Sits cleanly between fusion and generation; opt-in per config.

### Ingestion

**Pluggable ingestion methods.** Vector, document, and graph ingestion all implement `BaseIngestionMethod`. Required vs optional methods are part of the contract: required-method failures abort the ingest; optional methods log and continue. Each method is self-contained and dispatches generically through `IngestionService`.

**Drawing-aware ingestion for diagram-first documents.** Schematics, P&ID, wiring, and mechanical drawings break in chunk-and-pray pipelines — page 2 loses its connection to page 5. The drawing pipeline takes a different path: a vision LLM extracts structure once per page (components, labels, off-page connector tags) and emits every cross-page reference into the graph store as an edge candidate. Cross-sheet reasoning happens at query time, *over the assembled graph*, by the model itself. DXF files parse natively through `ezdxf` with no LLM calls. Symbol vocabularies (IEC 60617 + ISA 5.1 ship as defaults) are consumer-overridable. Per-page vision failures soft-skip the page and record a note rather than aborting the whole ingest.

**Index-time enrichment.** Three orthogonal opt-ins:
- `chunk_context_headers` — templated `Document: X | Type: Y | Page: N | Section: …` prepended to chunk text.
- `DocumentExpansionConfig` — LLM generates synthetic queries each chunk could answer (docT5query-style); flows into both BM25 and embeddings.
- `ContextualChunkConfig` — Anthropic's Contextual Retrieval recipe; LLM generates a 50–100 token blob situating each chunk within its source document, prepended before embedding/BM25.

### Generation

**Grounding gate.** Before the final LLM call, a relevance gate scores retrieved context against the query and refuses to answer when context is irrelevant rather than hallucinating from low-confidence chunks.

**Lost-in-the-middle mitigation.** Generation context can be assembled in score-descending order (default), primacy-recency, or sandwich. The non-default orderings put high-confidence chunks where U-shaped attention actually uses them.

**Long-context direct generation.** When the corpus fits the model's window, `FULL_CONTEXT` mode loads the full corpus into a stable prompt prefix optimized for prompt-cache hits. Pairs cleanly with `AUTO` routing for transparent retrieval-or-direct dispatch.

### Observability

**Per-source health view.** `Source.fully_ingested` plus `Source.ingestion_notes: list[str]` (entries formatted `<step>:<level>:<reason>`) record any non-fatal degradation that happened during ingest — a contextualization skip on an oversized document, a vision page failure, a graph-extraction partial. `KnowledgeManager.health(source_id)` fuses these with `SourceStats` (retrieval hits, grounded vs ungrounded answer counts) and the `stale` flag (embedding-model migration). The CLI surfaces it as `rfnry-rag knowledge inspect <source_id>`.

**Per-query trace.** Pass `trace=True` to receive a `RetrievalTrace` capturing rewritten queries, per-method results (keyed by method name, including empty-result methods), fusion output, reranking, grounding decision, routing decision, and per-stage timings. Default `trace=False` is byte-for-byte unchanged.

**Benchmark harness.** Structured test cases run through `RagEngine.benchmark()` or the CLI. Aggregates exact match, F1, retrieval recall and precision (when expected source IDs are provided), and optional LLM-judge scores. Per-case traces are part of the report so individual failures are debuggable.

**Structured event stream.** `Observability` on `RagEngineConfig` is always-on. Every entry point, every LLM call, and every retrieval / ingestion method emit a typed `ObservabilityRecord` (Pydantic, JSON-serializable) through the configured `Sink`. Default `JsonlStderrSink` writes one JSON line per record; swap to `JsonlFileSink`, `MultiSink`, or a custom `Sink` (an OTel adapter is ~25 lines) without touching pipeline code. `kind` discriminates events: `query.start`, `provider.call`, `retrieval.method.success`, etc. Pass `Observability(sink=NullSink())` to silence.

**Row-per-transaction telemetry.** `Telemetry` writes one `QueryTelemetryRow` per query and one `IngestTelemetryRow` per ingest. Rows carry `outcome`, `duration_ms`, per-method timings, raw token counts (input / output / cache_creation / cache_read), and routing/grounding decisions. Default sink is stderr; use `SqlAlchemyTelemetrySink(metadata_store)` to persist into `rag_query_telemetry` / `rag_ingest_telemetry` tables for admin-UI consumption. Cost is the consumer's stack — the library emits raw token counts only.

### Providers

**Provider-agnostic facades.** `Embeddings`, `Vision`, `Reranking`, and `LanguageModelClient` dispatch to the correct backend at runtime — Anthropic, OpenAI, Gemini, Voyage, Cohere — based on the configured `LanguageModel`. The retrieval pipeline looks identical regardless of which model is wired in; swapping providers is a configuration change, not a code change.

**Native SDK + BAML hybrid.** Plain text generation and streaming go through native provider SDKs in `providers/text_generation.py` (per-backend dispatch with prompt-cache-aware blocks for Anthropic). Structured-output calls — vision page analysis, entity extraction, document synthesis, the answer-quality judge — go through BAML for schema-typed parsing, retry policies, and primary-plus-fallback provider routing.

**Prompt-injection-resistant fencing.** Every user-controlled prompt parameter is wrapped with explicit start/end markers and a "treat as untrusted" directive. A contract test scans every BAML source file and fails CI if any function ships a user-input parameter unfenced. A second contract test scans for banned domain vocabulary and fails CI if any leak into prompts.

---

## License

MIT — see [LICENSE](../../LICENSE).
