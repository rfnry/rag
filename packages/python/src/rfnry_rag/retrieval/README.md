# rfnry-rag — Retrieval SDK

Composable retrieval-augmented generation. Ingest documents, search them, generate answers.

For setup, environment variables, and observability see the [main README](../../README.md). For concepts and architecture see the [retrieval documentation](../../docs/retrieval.md).

## RagEngine

```python
from rfnry_rag.retrieval import (
    RagEngine, RagServerConfig,
    PersistenceConfig, IngestionConfig, RetrievalConfig, GenerationConfig,
    TreeIndexingConfig, TreeSearchConfig,
    QdrantVectorStore, SQLAlchemyMetadataStore,
    PostgresDocumentStore, Neo4jGraphStore,
    Embeddings, FastEmbedSparseEmbeddings, Vision,
    Reranking, HyDeRewriting,
    LanguageModelClient, LanguageModelProvider,
)

rewriter_lm = LanguageModelClient(
    provider=LanguageModelProvider(
        provider="anthropic", model="claude-haiku-4-5-20251001", api_key="...",
    ),
    max_tokens=512,
    temperature=0.3,
)

config = RagServerConfig(
    persistence=PersistenceConfig(
        vector_store=QdrantVectorStore(url="http://localhost:6333", collection="docs"),
        metadata_store=SQLAlchemyMetadataStore(url="postgresql+asyncpg://user:pass@localhost/rag"),
        document_store=PostgresDocumentStore(url="postgresql+asyncpg://user:pass@localhost/rag"),
        graph_store=Neo4jGraphStore(uri="bolt://localhost:7687", username="neo4j", password="..."),
    ),
    ingestion=IngestionConfig(
        embeddings=Embeddings(LanguageModelProvider(provider="openai", model="text-embedding-3-small", api_key="...")),
        sparse_embeddings=FastEmbedSparseEmbeddings(),
        vision=Vision(LanguageModelProvider(provider="anthropic", model="claude-sonnet-4-20250514", api_key="...")),
        chunk_size=500,
        chunk_overlap=50,
        parent_chunk_size=1500,
        contextual_chunking=True,
        dpi=300,
    ),
    retrieval=RetrievalConfig(
        top_k=5,
        reranker=Reranking(LanguageModelProvider(provider="cohere", model="rerank-v3.5", api_key="...")),
        query_rewriter=HyDeRewriting(lm_client=rewriter_lm),
        source_type_weights={"manual": 1.0, "transcript": 0.5},
        parent_expansion=True,
        cross_reference_enrichment=True,
        enrich_lm_client=LanguageModelClient(
            provider=LanguageModelProvider(provider="openai", model="gpt-4o", api_key="..."),
        ),
    ),
    generation=GenerationConfig(
        lm_client=LanguageModelClient(
            provider=LanguageModelProvider(
                provider="anthropic", model="claude-sonnet-4-20250514", api_key="...",
            ),
        ),
        grounding_enabled=True,
        grounding_threshold=0.5,
        relevance_gate_enabled=True,
        relevance_gate_model=LanguageModelClient(
            provider=LanguageModelProvider(
                provider="anthropic", model="claude-haiku-4-5-20251001", api_key="...",
            ),
        ),
        guiding_enabled=True,
    ),
)
```

At least one retrieval path must be configured: vector (`vector_store` + `embeddings`), document (`document_store`), or graph (`graph_store`). Everything else is optional — add stores, search paths, and quality gates as needed.

---

## Modules

### Pipeline Methods

After initialization, `RagEngine` exposes configured pipeline methods via namespace properties. Use these for fine-grained control over individual retrieval and ingestion paths.

```python
async with RagEngine(config) as rag:
    # Attribute access — call individual methods directly
    chunks = await rag.retrieval.vector.search("query", top_k=20)
    doc_chunks = await rag.retrieval.document.search("query", top_k=10)

    # Conditional fallback
    if len(chunks) < 5 and "document" in rag.retrieval:
        chunks.extend(await rag.retrieval.document.search("query", top_k=10))

    # Iterate all configured methods
    for method in rag.retrieval:
        results = await method.search("query", top_k=10)
        print(f"{method.name}: {len(results)} results")

    # Check what's configured
    print("vector" in rag.retrieval)   # True if vector_store + embeddings configured
    print("document" in rag.retrieval) # True if document_store configured
    print("graph" in rag.retrieval)    # True if graph_store configured

    # Feed hand-picked results into generation
    answer = await rag.generate(query="What is GDPR?", chunks=chunks)
```

Methods are self-contained: each handles its own errors (returns empty list on failure) and logs with hierarchical prefixes (`retrieval.methods.vector`, `retrieval.methods.document`).

### Ingestion

Parse files, split into semantic text chunks, embed, store.

```python
async with RagEngine(config) as rag:
    # PDF, text, markdown
    source = await rag.ingest("manual.pdf", knowledge_id="helios", source_type="manual")

    # Images (requires vision provider)
    source = await rag.ingest("diagram.png", knowledge_id="helios")

    # Raw text
    source = await rag.ingest_text("Q: What oil? A: SAE 30.", knowledge_id="helios")

    # Page range and resume support
    source = await rag.ingest("large.pdf", knowledge_id="helios", page_range="1-50")

    # Tree indexing for structured documents (PDF only)
    source = await rag.ingest("annual_report.pdf", knowledge_id="reports", tree_index=True)
```

`ingest()` auto-routes by file extension. When `tree_index=True`, a hierarchical tree index is built alongside chunks — the SDK detects the document's table of contents, maps sections to page ranges, and generates summaries for LLM-based tree search at query time. Documents (`.pdf`, `.txt`, `.md`, images) go through chunking. Technical files (`.l5x`, `.xml`) go through the analyze pipeline when a metadata store is configured.

#### Analyze pipeline

Three-phase ingestion for technical documents. Extract entities, discover cross-page relationships, then embed.

```python
async with RagEngine(config) as rag:
    # Phase 1: per-page analysis (vision LLM for PDFs, deterministic for L5X/XML)
    source = await rag.analyze("schematic.pdf", knowledge_id="plant-1")
    # source.status == "analyzed"

    # Phase 2: cross-page relationship discovery
    source = await rag.synthesize(source.source_id)
    # source.status == "synthesized"

    # Phase 3: embed and store
    source = await rag.complete_ingestion(source.source_id)
    # source.status == "completed"
```

Or let `ingest()` run all three phases automatically for supported extensions.

#### Batch ingestion

Bulk text record ingestion with batched embedding, duplicate detection, and progress callbacks. Standalone from `RagEngine`.

```python
from rfnry_rag.retrieval.modules.ingestion.chunk.batch import (
    BatchIngestionService, BatchConfig, TextRecord,
)

service = BatchIngestionService(
    embeddings=embeddings,
    vector_store=vector_store,
    embedding_model_name="openai:text-embedding-3-small",
    config=BatchConfig(batch_size=100, concurrency=5, skip_duplicates=True),
    metadata_store=metadata_store,
)

records = [
    TextRecord(text="Document content...", title="doc-1", knowledge_id="corpus"),
    TextRecord(text="Another document...", title="doc-2", knowledge_id="corpus"),
]
stats = await service.ingest_batch(records)
print(f"{stats.succeeded}/{stats.total} in {stats.duration_seconds:.1f}s")
```

### Retrieval

The retrieval pipeline has three stages: **query rewriting** (pre-retrieval), **multi-path search**, and **post-retrieval refinement**.

#### Query rewriting (pre-retrieval)

Before any search runs, the SDK can optionally rewrite the query to improve retrieval quality. This adds one LLM call per query — an opt-in cost-quality tradeoff. Three strategies are available:

- **HyDE** — Generates a hypothetical answer and searches using that embedding instead of the question's. Most effective when question language and document language differ (common in technical domains).
- **Multi-query** — Generates 2-3 query variants that capture different phrasings of the same intent. All are searched, results fused.
- **Step-back** — Generates a broader version of the query to retrieve background context the specific query would miss.

When enabled, rewriting runs before all search paths. The rewritten queries feed into the same pipeline — vector, keyword, document, and graph all benefit from the improved queries.

#### Search paths

Up to five search paths run concurrently per query and merge results via reciprocal rank fusion:

- **Vector** — Dense semantic similarity (always active). When `sparse_embeddings` is configured, SPLADE sparse vectors are stored alongside dense vectors and Qdrant runs hybrid search (dense + sparse) in a single query, combining semantic understanding with exact term matching.
- **BM25** — In-memory BM25 ranking, runs inside `VectorRetrieval` when `bm25_enabled=True`. Automatically disabled when `sparse_embeddings` is configured.
- **Document** — Full-text ranked search + substring matching on original documents (requires document store).
- **Graph** — Entity full-text lookup + N-hop relationship traversal (requires graph store).
- **Enrich** — Structured retrieval with entity field filtering and cross-reference enrichment (requires metadata store).
- **Tree** — LLM reasoning over hierarchical document structure (requires metadata store + `TreeSearchConfig.enabled`). The LLM navigates the document's section tree via a BAML tool-use loop, fetching pages and drilling into subtrees to find relevant content. Best for long structured documents where section-level retrieval matters.

#### Post-retrieval

After fusion, two optional stages run in sequence:

- **Reranking** — Cross-encoder reranking against the original query via `Reranking` (Cohere, Voyage, or LLM-based). Reorders fused results by relevance before truncating to `top_k`.
- **Chunk refinement** — Extractive (context window extraction) or abstractive (LLM-based summarization) refinement of the final chunks.

```python
async with RagEngine(config) as rag:
    chunks = await rag.retrieve("part number 8842-A", knowledge_id="helios")
    for chunk in chunks:
        print(f"[{chunk.score:.2f}] {chunk.content[:100]}...")
```

### Generation

Full query pipeline: retrieval, score gate, LLM relevance gate, optional clarification, and LLM generation.

```python
async with RagEngine(config) as rag:
    # Single query
    result = await rag.query("How do I replace the filter?", knowledge_id="helios")
    print(result.answer, result.grounded, result.confidence)

    # Multi-turn conversation
    follow_up = await rag.query(
        "What about the oil filter?",
        knowledge_id="helios",
        history=[("How do I replace the filter?", result.answer or "")],
    )

    # Streaming
    async for event in rag.query_stream("How do I replace the filter?", knowledge_id="helios"):
        if event.type == "chunk":
            print(event.content, end="", flush=True)
        elif event.type == "sources":
            for ref in event.sources:
                print(f"\n  - {ref.name} (page {ref.page_number})")
        elif event.type == "done":
            print(f"\n[grounded={event.grounded}, confidence={event.confidence:.2f}]")
```

When grounding gates reject a query, `result.grounded` is `False` and `result.answer` contains an escalation message. When guiding is enabled, the relevance gate may return a `result.clarification` with a question and options.

### Knowledge

Source CRUD, chunk inspection, statistics, and embedding migration detection.

```python
async with RagEngine(config) as rag:
    # List sources
    sources = await rag.knowledge.list(knowledge_id="helios")

    # Inspect a source
    source = await rag.knowledge.get(source_id)
    chunks = await rag.knowledge.get_chunks(source_id)
    stats = await rag.knowledge.get_stats(source_id)

    # Remove a source and all its vectors
    deleted = await rag.knowledge.remove(source_id)
```

Embedding migration is checked at startup. If the configured embedding model differs from what was used to embed existing sources, those sources are flagged as stale.

### Evaluation

Evaluate retrieval and generation quality with configurable metrics.

- **ExactMatch** — Binary match after normalization
- **F1Score** — Token-level overlap
- **LLMJudgment** — LLM-as-judge scoring via BAML
- **RetrievalPrecision / RetrievalRecall** — Measure retrieval quality against ground truth

---

## Persistence

Four stores, all optional individually:

| Store | Purpose | Required |
|-------|---------|----------|
| **Vector store** | Chunk embeddings for semantic search | No (needs embeddings) |
| **Metadata store** | Source tracking, stats, analyze pipeline | No |
| **Document store** | Full document text for substring search | No |
| **Graph store** | Entity relationships for graph traversal | No |

At least one of vector, document, or graph must be configured.

## Providers

All providers use Python `Protocol` (structural typing). Swap freely, or implement your own — no inheritance required.

| Category | Options |
|----------|---------|
| Embeddings | `Embeddings(LanguageModelProvider(...))` — OpenAI, Voyage, Cohere |
| Sparse Embeddings | `FastEmbedSparseEmbeddings` (SPLADE via FastEmbed) |
| Generation | Via `LanguageModelClient` (any BAML-supported provider) |
| Reranking | `Reranking(LanguageModelProvider(...))` — Cohere, Voyage, or LLM-based |
| Query Rewriting | `HyDeRewriting`, `MultiQueryRewriting`, `StepBackRewriting` |
| Vision | `Vision(LanguageModelProvider(...))` — Anthropic, OpenAI |
| Vector Store | `QdrantVectorStore` (dense + sparse named vectors, hybrid search) |
| Metadata Store | `SQLAlchemyMetadataStore` (PostgreSQL via asyncpg, SQLite via aiosqlite) |
| Document Store | `PostgresDocumentStore`, `FilesystemDocumentStore` |
| Graph Store | `Neo4jGraphStore` (optional `neo4j>=5.0` dependency) |

---

## CLI

Install with `uv add "rfnry-rag[cli]"`. Configure once, use from anywhere.

```bash
rfnry-rag retrieval init                                    # Create config + .env templates
rfnry-rag retrieval status                                  # Verify config and connections

rfnry-rag retrieval ingest manual.pdf -k equipment          # Ingest a file
rfnry-rag retrieval ingest report.pdf -k reports --tree-index  # Ingest with tree index
rfnry-rag retrieval ingest --text "..." -k equipment        # Ingest raw text
rfnry-rag retrieval retrieve "part number RV-2201"          # Retrieval only, raw chunks
rfnry-rag retrieval retrieve "pressure" --min-score 0.4     # Filter low-quality results
rfnry-rag retrieval query "how to replace the filter?"      # Retrieve + generate answer
rfnry-rag retrieval query "what oil?" --session ticket-123  # Multi-turn with session context

rfnry-rag retrieval session list                            # List active sessions
rfnry-rag retrieval session clear ticket-123                # Clear a session

rfnry-rag retrieval knowledge list -k equipment             # List sources
rfnry-rag retrieval knowledge get <source-id>               # Source details
rfnry-rag retrieval knowledge chunks <source-id>            # Inspect chunks
rfnry-rag retrieval knowledge stats <source-id>             # Hit statistics
rfnry-rag retrieval knowledge remove <source-id>            # Delete source + chunks
```

Output auto-detects TTY: terminal gets human-readable, pipes get JSON. Override with `--json` or `--pretty`.

Config lives in `~/.config/rfnry_rag/config.toml` + `.env`. API keys in `.env`, providers and store URLs in the TOML. Run `rfnry-rag retrieval init` to see all available options.

See `examples/retrieval/cli/` for complete walkthroughs.

---

## API Reference

### `RagEngine`

Async context manager: `async with RagEngine(config) as rag:`

| Method | Returns | Description |
|--------|---------|-------------|
| `ingest(path, knowledge_id?, source_type?, metadata?, page_range?, tree_index?)` | `Source` | Ingest file (auto-routes by extension) |
| `ingest_text(content, knowledge_id?, source_type?, metadata?)` | `Source` | Ingest raw text |
| `analyze(path, knowledge_id?, source_type?, metadata?, page_range?)` | `Source` | Analyze pipeline phase 1 |
| `synthesize(source_id)` | `Source` | Analyze pipeline phase 2 |
| `complete_ingestion(source_id)` | `Source` | Analyze pipeline phase 3 |
| `query(text, knowledge_id?, history?, min_score?)` | `QueryResult` | Full RAG pipeline |
| `query_stream(text, knowledge_id?, history?, min_score?)` | `AsyncIterator[StreamEvent]` | Streaming RAG pipeline |
| `retrieve(text, knowledge_id?, min_score?)` | `list[RetrievedChunk]` | Retrieval only (no generation) |
| `embed(texts)` | `list[list[float]]` | Raw embedding (batch) |
| `embed_single(text)` | `list[float]` | Raw embedding (single) |
| `knowledge.list(knowledge_id?)` | `list[Source]` | List sources |
| `knowledge.get(source_id)` | `Source \| None` | Get source |
| `knowledge.get_chunks(source_id)` | `list[Chunk]` | Inspect chunks |
| `knowledge.get_stats(source_id)` | `SourceStats \| None` | Hit statistics |
| `knowledge.remove(source_id)` | `int` | Delete source + vectors |

### Config Reference

#### `RagServerConfig`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `persistence` | `PersistenceConfig` | yes | Stores |
| `ingestion` | `IngestionConfig` | yes | Embeddings + chunking |
| `retrieval` | `RetrievalConfig` | no | Search tuning |
| `generation` | `GenerationConfig` | no | LLM generation |
| `tree_indexing` | `TreeIndexingConfig` | no | Tree index building |
| `tree_search` | `TreeSearchConfig` | no | Tree-based search |

#### `PersistenceConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `vector_store` | `BaseVectorStore` | `None` | Vector store for embeddings |
| `metadata_store` | `BaseMetadataStore` | `None` | Source metadata, stats, analyze pipeline |
| `document_store` | `BaseDocumentStore` | `None` | Full document text for substring search |
| `graph_store` | `BaseGraphStore` | `None` | Entity-relationship graph (Neo4j) |

#### `IngestionConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `embeddings` | `BaseEmbeddings` | `None` | Embedding provider (required for vector path) |
| `vision` | `BaseVision` | `None` | Vision provider for images and PDF analysis |
| `chunk_size` | `int` | `500` | Target tokens per chunk |
| `chunk_overlap` | `int` | `50` | Token overlap between chunks |
| `parent_chunk_size` | `int` | `0` | Parent chunk size (0 = disabled) |
| `parent_chunk_overlap` | `int` | `200` | Token overlap between parent chunks |
| `contextual_chunking` | `bool` | `True` | Prepend document context to each chunk before embedding |
| `sparse_embeddings` | `BaseSparseEmbeddings` | `None` | SPLADE sparse vectors for hybrid search |
| `lm_client` | `LanguageModelClient` | `None` | LLM config for structured analysis |
| `dpi` | `int` | `300` | PDF rendering resolution (analyze pipeline) |

#### `RetrievalConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `top_k` | `int` | `5` | Results returned |
| `reranker` | `BaseReranking` | `None` | Cross-encoder reranking |
| `query_rewriter` | `BaseQueryRewriting` | `None` | Pre-retrieval query rewriting |
| `parent_expansion` | `bool` | `True` | Return parent chunks when child chunks match |
| `bm25_enabled` | `bool` | `False` | In-memory BM25 (deprecated when sparse_embeddings configured) |
| `source_type_weights` | `dict[str, float]` | `None` | Score multipliers by source type |
| `cross_reference_enrichment` | `bool` | `True` | Fetch cross-referenced pages (analyze pipeline) |
| `enrich_lm_client` | `LanguageModelClient` | `None` | LLM for query analysis |
| `chunk_refiner` | `BaseChunkRefinement` | `None` | Post-retrieval chunk refinement |

#### `GenerationConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `lm_client` | `LanguageModelClient` | `None` | LLM config. Required for `query()` |
| `system_prompt` | `str` | default | System prompt for generation |
| `grounding_enabled` | `bool` | `False` | Score-based grounding gate |
| `grounding_threshold` | `float` | `0.5` | Minimum retrieval score (0-1) |
| `relevance_gate_enabled` | `bool` | `False` | LLM relevance gate (requires `grounding_enabled`) |
| `relevance_gate_model` | `LanguageModelClient` | `None` | LLM for relevance judgment |
| `guiding_enabled` | `bool` | `False` | Clarification questions (requires `relevance_gate_enabled`) |
| `step_lm_client` | `LanguageModelClient` | `None` | LLM for step-by-step reasoning |

#### `TreeIndexingConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | `bool` | `False` | Enable tree index building during ingestion |
| `model` | `LanguageModelClient` | `None` | LLM for TOC detection, structure extraction, summaries |
| `toc_scan_pages` | `int` | `20` | Pages to scan for table of contents |
| `max_pages_per_node` | `int` | `10` | Max pages before a node is split |
| `max_tokens_per_node` | `int` | `20000` | Max tokens before a node is split |
| `generate_summaries` | `bool` | `True` | Generate section summaries for navigation |
| `generate_description` | `bool` | `True` | Generate one-line document description |

#### `TreeSearchConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | `bool` | `False` | Enable tree-based search at query time |
| `model` | `LanguageModelClient` | `None` | LLM for reasoning-based tree traversal |
| `max_steps` | `int` | `5` | Max iterations in the tool-use loop |
| `max_context_tokens` | `int` | `50000` | Budget for accumulated page content |

#### `LanguageModelClient`

```python
from rfnry_rag.common.language_model import LanguageModelClient, LanguageModelProvider

# Simple — single provider
config = LanguageModelClient(
    provider=LanguageModelProvider(provider="openai", model="gpt-4o", api_key="..."),
)

# With fallback — auto-routes to fallback provider on failure
config = LanguageModelClient(
    provider=LanguageModelProvider(provider="anthropic", model="claude-haiku-4-5-20251001", api_key="..."),
    fallback=LanguageModelProvider(provider="openai", model="gpt-4o-mini", api_key="..."),
    strategy="fallback",
    max_retries=3,
)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `provider` | `LanguageModelProvider` | required | Primary LLM provider |
| `fallback` | `LanguageModelProvider` | `None` | Fallback LLM provider |
| `max_retries` | `int` | `3` | Retry attempts per provider (0-5) |
| `strategy` | `"primary_only" \| "fallback"` | `"primary_only"` | Routing strategy |
| `max_tokens` | `int` | `4096` | Max output tokens |
| `temperature` | `float` | `0.0` | Sampling temperature |
| `boundary_api_key` | `str` | `None` | Boundary proxy API key |

| Provider Field | Type | Default | Description |
|------|------|---------|-------------|
| `provider` | `str` | required | Provider name (`"openai"`, `"anthropic"`, etc.) |
| `model` | `str` | required | Model identifier |
| `api_key` | `str` | `None` | API key (falls back to environment variable) |

