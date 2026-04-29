# rfnry-rag

A retrieval and reasoning toolkit for Python. Two SDKs in one package: a configurable retrieval engine with multi-path search, query routing, and grounded generation; and a set of standalone reasoning services for analysis, classification, clustering, compliance, and evaluation. Both share a common protocol layer so consumers can swap embeddings, vector stores, rerankers, and language model providers without touching the engine.

## Get Started

Install with [uv](https://docs.astral.sh/uv/):

```bash
uv add rfnry-rag                  # core SDK
uv add "rfnry-rag[graph]"         # + Neo4j graph support
uv add "rfnry-rag[cli]"           # + command-line interface
```

A minimal retrieval setup:

```python
from rfnry_rag.retrieval import (
    RagEngine, RagServerConfig, PersistenceConfig, IngestionConfig,
    QdrantVectorStore, Embeddings,
)
from rfnry_rag import LanguageModelProvider

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

Reasoning services are standalone — instantiate one, configure it, call it:

```python
from rfnry_rag.reasoning import AnalysisService, AnalysisConfig, DimensionDefinition
from rfnry_rag import LanguageModelClient, LanguageModelProvider

lm = LanguageModelClient(provider=LanguageModelProvider(
    provider="anthropic", model="claude-sonnet-4-20250514", api_key="...",
))
analyzer = AnalysisService(lm_client=lm)
result = await analyzer.analyze(
    "My order FB-12345 hasn't arrived and I need it by Friday.",
    config=AnalysisConfig(
        dimensions=[DimensionDefinition("urgency", "How time-sensitive", "0.0-1.0")],
        summarize=True,
    ),
)
```

CLI mirrors the SDK surface (`rfnry-rag retrieval ...` / `rfnry-rag reasoning ...`).

---

## Features

### Modular hybrid retrieval

Retrieval methods are pluggable via the `BaseRetrievalMethod` protocol. Vector (dense + SPLADE + BM25 fused internally), document (full-text + substring), graph (entity lookup + N-hop traversal), and structural tree navigation all run concurrently per query and merge through reciprocal rank fusion with per-method weights. No mandatory backend; configure only the paths you need. Per-method error isolation means one failing path does not break the others.

### Query routing modes

Each query can be answered through one of four modes: standard retrieval, full-corpus direct context (cheaper and more accurate at small corpus sizes when prompt caching is in play), hybrid SELF-ROUTE (RAG first, escalate to long-context only when an LLM judges the retrieved chunks insufficient), or auto (corpus-token threshold picks retrieval-vs-direct per query). The choice is per-query and reflected in the trace.

### Adaptive retrieval parameters

A query classifier (heuristic by default; opt-in LLM for ambiguous text) labels each query by complexity and shape. Top-k scales by complexity (simple queries retrieve fewer chunks, complex queries retrieve more), and per-method weights shift by query type (entity-relationship queries lean on graph; comparative queries lean on document/tree). When initial retrieval returns weak results, the engine retries with expanded parameters and optionally escalates to direct-context mode.

### Multi-hop iterative retrieval

For queries that chain across entities ("what nationality is the performer of song X?"), an opt-in iterative service decomposes the query into sequential sub-questions, retrieves each independently with the full pipeline, and accumulates findings across hops. The decomposer self-summarizes between hops to bound prompt growth. Gates on query type or LLM verdict so cheap queries stay on the cheap path. Falls back to direct-context mode when accumulated chunks remain weak.

### Hierarchical summarization retrieval

A consumer-triggered build clusters chunks under a knowledge scope, generates summaries for each cluster, and recurses up to a configurable depth — producing a tree of progressively more abstract representations. A sibling retrieval method searches the summary nodes alongside the leaf-chunk methods, so abstract or broad queries match summaries while specific queries match leaves. Atomic blue/green rebuilds with immediate garbage collection of stale trees.

### Document expansion at index time

For each chunk, an optional LLM call generates synthetic queries the chunk would answer. The synthetic queries flow into both BM25 indexing and embedding generation, bridging the user-vocabulary-vs-document-vocabulary gap that hurts both lexical and dense retrieval. Stored separately on the chunk for transparency. Independently gated for embedding vs BM25.

### Chunk-position-aware context assembly

Generation context can be assembled in score-descending order (default), primacy-recency (highest-scored chunks at the beginning and end of context), or sandwich. The non-default orderings mitigate the U-shaped attention effect where LLMs use information at the beginning and end of context more reliably than information in the middle.

### Drawing-aware ingestion

A sibling pipeline for diagram-first documents (schematics, P&ID, wiring, mechanical). PDFs go through a vision LLM with consumer-overridable symbol vocabularies (IEC 60617 + ISA 5.1 ship as defaults). DXF files parse deterministically through `ezdxf` with no LLM calls — modelspace plus all paperspace layouts in tab order, so multi-sheet drawings emit one page per layout. Cross-sheet connectivity (off-page connectors, fuzzy label merges) resolves deterministically with LLM residue only when unresolved candidates remain.

### Diagnostic trace, failure classification, and benchmark harness

Every query can return a `RetrievalTrace` capturing per-stage state: rewritten queries, per-method results (keyed by method name, including empty-result methods), fusion output, reranking, refinement, grounding decision, confidence, routing decision, and per-stage timings. A heuristic `classify_failure` function maps a failed trace to one of seven failure types (vocabulary mismatch, chunk boundary, scope miss, entity-not-indexed, low relevance, insufficient context, unknown). A benchmark harness runs structured test cases and aggregates EM, F1, retrieval recall/precision, optional LLM-judge scores, and the failure-type distribution. Available as both a Python API and CLI.

### Reasoning SDK

Standalone services that compose without a vector store: `AnalysisService` (intent + named dimensions + entity tracking), `ClassificationService` (LLM or hybrid kNN→LLM), `ClusteringService` (K-Means + HDBSCAN with optional LLM cluster labeling), `ComplianceService` (policy violation checking against reference documents), `EvaluationService` (similarity + LLM-judge scoring). Compose them sequentially through `Pipeline` or wire them into a retrieval flow.

### Structured LLM I/O via BAML

All LLM calls go through [BAML](https://docs.boundaryml.com/) for structured output parsing, retry and fallback policies, and primary-plus-fallback provider routing through `LanguageModelClient`. Every user-controlled prompt parameter is fenced with explicit start/end markers; a contract test guards the convention so new prompts can't slip through unfenced. Domain-neutral prompts by default — features needing vocabulary (entity types, relationship keywords, symbol libraries) expose consumer-overridable config. Observability through Boundary Studio or `baml_py.Collector`.

### Unified CLI

`rfnry-rag retrieval ...` and `rfnry-rag reasoning ...` mirror the SDK surface for scripting, inspection, and one-off operations: ingest, query, retrieve (without generation), benchmark, analyze, classify, compliance-check. Configuration loads from `~/.config/rfnry_rag/config.toml` plus a `.env`.
