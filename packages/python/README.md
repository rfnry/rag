# rfnry-rag

A modular retrieval toolkit for Python. Compose vector, document, and graph retrieval methods into one pipeline, fuse their results, and route between indexed retrieval and full-context generation based on corpus size — automatically. Built around a single principle: as language models grow stronger and contexts grow longer, the toolkit gets out of their way instead of working around them.

## Get Started

Install with [uv](https://docs.astral.sh/uv/):

```bash
uv add rfnry-rag                  # core SDK
uv add "rfnry-rag[graph]"         # + Neo4j graph support
uv add "rfnry-rag[cli]"           # + command-line interface
```

A minimal retrieval setup:

```python
from rfnry_rag import RagEngine
from rfnry_rag.config import RagServerConfig, PersistenceConfig, IngestionConfig
from rfnry_rag.providers import Embeddings, LanguageModelProvider
from rfnry_rag.stores import QdrantVectorStore

config = RagServerConfig(
    persistence=PersistenceConfig(
        vector_store=QdrantVectorStore(url="http://localhost:6333", collection="docs"),
    ),
    ingestion=IngestionConfig(
        embeddings=Embeddings(LanguageModelProvider(
            provider="openai", model="text-embedding-3-small", api_key="...",
        )),
    ),
)

async with RagEngine(config) as rag:
    await rag.ingest("manual.pdf", knowledge_id="equipment")
    result = await rag.query("How do I replace the filter?", knowledge_id="equipment")
    print(result.answer)
```

CLI mirrors the SDK surface (`rfnry-rag ingest …` / `rfnry-rag query …` / `rfnry-rag benchmark …`).

---

## Features

### Retrieval

**Modular method composition.** Vector (dense + BM25 fused internally), document (Postgres FTS + substring), and graph (entity lookup + N-hop traversal). All paths are pluggable via the `BaseRetrievalMethod` protocol and run concurrently per query, merging through reciprocal rank fusion with per-method weights. No mandatory backend; configure only the paths you need. Per-method error isolation means one failing path does not break the others.

**Auto routing between indexed retrieval and full-context generation.** Each query is dispatched through one of three modes: `INDEXED` (the standard retrieval pipeline), `FULL_CONTEXT` (load the entire corpus into a prompt-cached prefix and let the model answer directly), or `AUTO` (a corpus-token threshold dispatches between them per query). As context windows grow, more corpora cross the threshold and transparently shift to `FULL_CONTEXT` without code changes.

**Cross-encoder reranking.** Optional reranking against the original query (Cohere, Voyage). Sits cleanly between fusion and generation; opt-in per config.

### Ingestion

**Pluggable ingestion methods.** Vector, document, and graph ingestion all implement `BaseIngestionMethod`. Required vs optional methods are part of the contract: required-method failures abort the ingest; optional methods log and continue. Each method is self-contained and dispatches generically through `IngestionService`.

**Drawing-aware ingestion for diagram-first documents.** Schematics, P&ID, wiring, and mechanical drawings are notoriously broken in chunk-and-pray pipelines — page 2 loses its connection to page 5, and the model is left guessing. The drawing pipeline takes a different path: a vision LLM extracts structure once per page (components, labels, off-page connector tags) and emits every cross-page reference into the graph store as an edge candidate. Cross-sheet reasoning then happens at query time, *over the assembled graph*, by the model itself. DXF files parse natively through `ezdxf` with no LLM calls — modelspace plus all paperspace layouts in tab order. Symbol vocabularies (IEC 60617 + ISA 5.1 ship as defaults) are consumer-overridable.

**Document expansion at index time.** For each chunk, an optional LLM call generates synthetic queries the chunk would answer. The synthetic queries flow into both BM25 indexing and embedding generation, bridging the user-vocabulary-vs-document-vocabulary gap that hurts both lexical and dense retrieval. Independently gated for embedding vs BM25.

### Generation

**Grounding gate.** Before the generation LLM call, a relevance gate checks whether the retrieved context actually pertains to the query. The engine refuses to answer when context is irrelevant rather than hallucinating from low-confidence chunks.

**Lost-in-the-middle mitigation.** Generation context can be assembled in score-descending order (default), primacy-recency (highest-scored chunks at the beginning and end), or sandwich. The non-default orderings put high-confidence chunks where U-shaped attention actually uses them.

**Long-context direct generation.** When the corpus fits the model's context window, `FULL_CONTEXT` mode loads the full corpus into a stable prompt prefix optimized for prompt-cache hits. Pairs cleanly with `AUTO` routing for transparent retrieval-or-direct dispatch.

### Observability

**Per-query trace.** Pass `trace=True` to receive a `RetrievalTrace` capturing the full per-stage state: rewritten queries, per-method results (keyed by method name, including empty-result methods), fusion output, reranking, grounding decision, routing decision, and per-stage timings. Default `trace=False` is byte-for-byte unchanged.

**Benchmark harness.** Structured test cases run through a Python API or CLI command. Aggregates exact match, F1, retrieval recall and precision (when expected source IDs are provided), and optional LLM-judge scores. Per-case traces are part of the report so individual failures are debuggable.

### Providers

**Provider-agnostic facades.** `Embeddings`, `Vision`, `Reranking`, and `LanguageModelClient` dispatch to the correct backend at runtime — Anthropic, OpenAI, Voyage, Cohere, Gemini — based on the configured `LanguageModelProvider`. The retrieval pipeline looks identical regardless of which model is wired in, so swapping providers is a configuration change, not a code change.

**Structured LLM I/O via BAML.** Every LLM call goes through [BAML](https://docs.boundaryml.com/) — schema-typed input/output, automatic retry and fallback policies, primary-plus-fallback provider routing through `LanguageModelClient`. Observability through Boundary Studio or `baml_py.Collector` for in-process token tracking.

**Prompt-injection-resistant fencing.** Every user-controlled prompt parameter is wrapped with explicit start/end markers and a "treat as untrusted" directive. A contract test scans every BAML source file and fails CI if any function ships a user-input parameter unfenced.

**Domain-neutral by default.** No hardcoded domain vocabulary lives in any prompt. Features that need vocabulary (entity types, relationship keywords, symbol libraries) expose consumer-overridable config with empty defaults. A second contract test scans for banned domain terms and fails CI if any leak in.
