# Plan: Fully Modular Retrieval & Ingestion Pipeline



## Context



x64rag has grown search methods (vector, chunk-BM25, document FTS, graph) and ingestion paths (vector embeddings, document storage, graph extraction, tree indexing) as separate concerns hard-coded into `RetrievalService` and `IngestionService`. This makes it difficult to use document-level search without embeddings, compose methods dynamically, or add new methods without touching the core orchestrator.



**Goal:** Formalize search and ingestion as plugin architectures. All methods conform to lightweight `Protocol`-based interfaces. The server assembles the active pipeline from what's configured. No mandatory vector DB or embeddings — at least one method must be activatable. Backward compatible.



**Naming convention:** Search methods are named by *what they operate on*, not the paradigm. Both `ChunkSearch` and `DocumentRetrieval` are lexical methods — the distinction is granularity, not technique.



- `KeywordSearch` → renamed to **`ChunkSearch`** (BM25 on chunks from vector store)

- New: **`DocumentRetrieval`** (FTS/BM25 on full documents from document store)

- `VectorRetrieval` and `GraphRetrieval` unchanged



Similarly for ingestion: `DocumentIngestion` (not `LexicalIngestion`).



---



## Architecture



```

┌──────────────────────────────────────────────────────────┐

│                       RagServer                          │

│                                                          │

│  initialize() builds both lists from config:             │

│                                                          │

│  ingestion_methods: list[BaseIngestionMethod]             │

│  ┌──────────────┬──────────────┬───────────┬──────────┐  │

│  │VectorIngest° │DocumentIngest│GraphIngest│TreeIngest│  │

│  └──────┬───────┴──────┬───────┴─────┬─────┴────┬─────┘  │

│         │              │             │          │         │

│    vector_store   document_store  graph_store  tree_svc   │

│    + embeddings                   + lm_config             │

│                                                          │

│  retrieval_methods: list[BaseRetrievalMethod]                   │

│  ┌────────────────────────┬──────────────┬───────────┐   │

│  │   VectorRetrieval°        │DocumentRetrieval│GraphRetrieval│   │

│  │ (dense + SPLADE + BM25)│              │           │   │

│  └────────────┬───────────┴──────┬───────┴─────┬─────┘   │

│              │                    │             │         │

│         vector_store        document_store  graph_store   │

│         + embeddings                                      │

│                                                          │

│  ° = only if embeddings + vector_store configured         │

│                                                          │

│  IngestionService(chunker, ingestion_methods, ...)        │

│  RetrievalService(retrieval_methods, reranker, ...)          │

└──────────────────────────────────────────────────────────┘

```



**Key insight:** VectorRetrieval is not a single strategy—it's a *family of chunk-level strategies*:

- Dense vector search (always)

- Hybrid dense + SPLADE (if sparse_embeddings configured)

- BM25 on chunks (if bm25_enabled and not sparse_embeddings)



VectorRetrieval runs these internally, uses RRF to fuse them, returns one `list[RetrievedChunk]`. This keeps chunk-level strategies bundled (they're all chunk-scoped retrieval) while DocumentRetrieval and GraphRetrieval operate at their own scope levels.



---



## Phase 1: Search Protocol & Methods



### `BaseRetrievalMethod` Protocol (NEW)

**File:** `src/rfnry_rag/retrieval/modules/retrieval/base.py`



```python

class BaseRetrievalMethod(Protocol):

    @property

    def name(self) -> str: ...



    async def search(

        self,

        query: str,

        top_k: int,

        filters: dict[str, Any] | None = None,

        knowledge_id: str | None = None,

    ) -> list[RetrievedChunk]: ...

```



### `DocumentRetrieval` (NEW)

**File:** `src/rfnry_rag/retrieval/modules/retrieval/methods/document.py`



Wraps `BaseDocumentStore.search_content()`. Converts `ContentMatch` → `RetrievedChunk`. Logic extracted from current `RetrievalService._content_matches_to_chunks()`.



### `GraphRetrieval` (NEW)

**File:** `src/rfnry_rag/retrieval/modules/retrieval/methods/graph.py`



Wraps `BaseGraphStore.query_graph()`. Converts `GraphResult` → `RetrievedChunk`. Logic extracted from current `RetrievalService._graph_results_to_chunks()`.



### Refactor `VectorRetrieval` — absorb `KeywordSearch` internally

**File:** `src/rfnry_rag/retrieval/modules/retrieval/methods/vector.py`



Add `bm25_enabled` and `bm25_max_indexes` parameters. `VectorRetrieval.search()` now internally:

1. Runs dense search (or hybrid if SPLADE)

2. If `bm25_enabled`, also runs BM25 on chunks (code extracted from `KeywordSearch`)

3. Uses RRF to fuse results from all enabled strategies

4. Returns single `list[RetrievedChunk]`



The BM25 cache, invalidation, and LRU eviction logic all move from `KeywordSearch` into `VectorRetrieval` as private internals.



**Keep:** `KeywordSearch` class stays for backward compatibility if any code imports it directly, but it's no longer wired in the server or tests.



### Add `name` property

- `VectorRetrieval`: add `name = "vector"` property (protocol requirement)



### `VectorRetrieval.search()` signature alignment

Align with `BaseRetrievalMethod` protocol: `search(query, top_k, filters=None, knowledge_id=None)`.



---



## Phase 2: Refactor RetrievalService



**File:** `src/rfnry_rag/retrieval/modules/retrieval/search/service.py`



Replace hardcoded dependencies with `retrieval_methods: list[BaseRetrievalMethod]`.



```python

class RetrievalService:

    def __init__(

        self,

        retrieval_methods: list[BaseRetrievalMethod],

        reranking: BaseReranking | None = None,

        top_k: int = 5,

        source_type_weights: dict[str, float] | None = None,

        query_rewriter: BaseQueryRewriter | None = None,

        chunk_refiner: BaseChunkRefiner | None = None,

    ) -> None: ...

```



`_search_single_query()` becomes:

```python

named_tasks = {

    method.name: method.search(query=query, top_k=fetch_k, filters=filters, knowledge_id=knowledge_id)

    for method in self._retrieval_methods

}

gathered = await asyncio.gather(*named_tasks.values())

result_lists = [list(r) for r in gathered]

```



Remove `_content_matches_to_chunks()` and `_graph_results_to_chunks()` — those conversions move into `DocumentRetrieval` and `GraphRetrieval`. Remove `_keyword_search` reference.



Keep `_apply_source_weights()` and `_build_filters()`. Tree chunks still injected pre-RRF via `tree_chunks` parameter on `retrieve()`.



---



## Phase 3: Ingestion Protocol & Methods



### `BaseIngestionMethod` Protocol (NEW)

**File:** `src/rfnry_rag/retrieval/modules/ingestion/base.py`



```python

class BaseIngestionMethod(Protocol):

    @property

    def name(self) -> str: ...



    async def ingest(

        self,

        source_id: str,

        knowledge_id: str | None,

        source_type: str | None,

        source_weight: float,

        title: str,

        full_text: str,

        chunks: list[ChunkedContent],

        tags: list[str],

        metadata: dict[str, Any],

        hash_value: str | None = None,

        pages: list[ParsedPage] | None = None,

    ) -> None: ...



    async def delete(self, source_id: str) -> None: ...

```



### `VectorIngestion` (NEW — extracted from `IngestionService`)

**File:** `src/rfnry_rag/retrieval/modules/ingestion/methods/vector.py`



Extracted from `IngestionService._embed_and_store_incremental()` and `_build_points()`. Handles embed → build VectorPoint → upsert. Also owns `_embed_sparse_safe()`.



### `DocumentIngestion` (NEW)

**File:** `src/rfnry_rag/retrieval/modules/ingestion/methods/document.py`



Wraps `BaseDocumentStore.store_content()` / `delete_content()`. Replaces the `if self._document_store: ...` blocks in `IngestionService.ingest()` and `ingest_text()`.



### `GraphIngestion` (NEW)

**File:** `src/rfnry_rag/retrieval/modules/ingestion/methods/graph.py`



Wraps graph entity extraction + `BaseGraphStore`. Currently ad-hoc in `StructuredIngestionService`.



### `TreeIngestion` (NEW)

**File:** `src/rfnry_rag/retrieval/modules/ingestion/methods/tree.py`



Thin wrapper around existing `TreeIndexingService`. Only runs when `pages` are provided.



---



## Phase 4: Refactor IngestionService



**File:** `src/rfnry_rag/retrieval/modules/ingestion/chunk/service.py`



Replace direct store dependencies with `ingestion_methods: list[BaseIngestionMethod]`.



```python

class IngestionService:

    def __init__(

        self,

        chunker: SemanticChunker,

        ingestion_methods: list[BaseIngestionMethod],

        metadata_store: BaseMetadataStore | None = None,

        source_type_weights: dict[str, float] | None = None,

        vision_parser: BaseVision | None = None,

        contextual_chunking: bool = True,

        on_ingestion_complete: Callable[[str | None], Awaitable[None]] | None = None,

    ) -> None: ...

```



`ingest()` flow:

1. Parse file → pages

2. Chunk pages → `chunks`

3. Build full_text from pages

4. For each method: `await method.ingest(source_id, ..., full_text, chunks, pages, ...)`

5. Create source in metadata_store (if present)

6. Fire `on_ingestion_complete` callback



`ingest_text()` follows same pattern but skips file parsing.



`_check_duplicate()`, `_resolve_weight()` stay on `IngestionService` (orchestration concerns).



---



## Phase 5: Server Wiring



**File:** `src/rfnry_rag/retrieval/server.py`



### Config changes

```python

@dataclass

class PersistenceConfig:

    vector_store: BaseVectorStore | None = None      # was required

    metadata_store: BaseMetadataStore | None = None

    document_store: BaseDocumentStore | None = None

    graph_store: BaseGraphStore | None = None



@dataclass

class IngestionConfig:

    embeddings: BaseEmbeddings | None = None         # was required

    # ... rest unchanged

```



### `RagServer.initialize()` assembly

```python

ingestion_methods: list[BaseIngestionMethod] = []

retrieval_methods: list[BaseRetrievalMethod] = []



# Vector path

if persistence.vector_store and ingestion.embeddings:

    vector_size = await ingestion.embeddings.embedding_dimension()

    await persistence.vector_store.initialize(vector_size)

    ingestion_methods.append(VectorIngestion(persistence.vector_store, ingestion.embeddings, ingestion.sparse_embeddings))

    

    # VectorRetrieval now owns BM25 internally

    vector_search = VectorRetrieval(

        persistence.vector_store, 

        ingestion.embeddings, 

        ingestion.sparse_embeddings,

        bm25_enabled=retrieval.bm25_enabled and not ingestion.sparse_embeddings,

        bm25_max_indexes=retrieval.bm25_max_indexes,

        ...

    )

    retrieval_methods.append(vector_search)



# Document path

if persistence.document_store:

    await persistence.document_store.initialize()

    ingestion_methods.append(DocumentIngestion(persistence.document_store))

    retrieval_methods.append(DocumentRetrieval(persistence.document_store))



# Graph path

if persistence.graph_store:

    await persistence.graph_store.initialize()

    ingestion_methods.append(GraphIngestion(persistence.graph_store, ingestion.lm_config))

    retrieval_methods.append(GraphRetrieval(persistence.graph_store))



# Tree path

if tree_indexing.enabled and persistence.metadata_store:

    ingestion_methods.append(TreeIngestion(self._tree_indexing_service))



# Validation

if not ingestion_methods:

    raise ConfigurationError("At least one store must be configured")

```



### Backward compatibility

- Existing configs with `vector_store` + `embeddings` work identically

- `StructuredIngestionService` unchanged (still takes stores directly)

- `StructuredRetrievalService` unchanged (vector-only)

- Collection-scoped retrieval pipelines rebuilt with same pattern



---



## Critical Files



| File | Type | Change |

|------|------|--------|

| `modules/retrieval/base.py` | NEW | `BaseRetrievalMethod` protocol |

| `modules/retrieval/methods/__init__.py` | NEW | Package init |

| `modules/retrieval/methods/vector.py` | NEW | `VectorRetrieval` (absorbs `KeywordSearch` BM25 logic internally) |

| `modules/retrieval/methods/document.py` | NEW | `DocumentRetrieval` (wraps `BaseDocumentStore`) |

| `modules/retrieval/methods/graph.py` | NEW | `GraphRetrieval` (wraps `BaseGraphStore`) |

| `modules/retrieval/search/vector.py` | DELETE | Old `VectorSearch` — moved to `methods/vector.py` as `VectorRetrieval` |

| `modules/retrieval/search/keyword.py` | DEPRECATED | Keep for backward compatibility if directly imported, but unused by server |

| `modules/retrieval/search/service.py` | REFACTOR | `retrieval_methods: list[BaseRetrievalMethod]`, generic dispatch, remove hardcoded branches |

| `modules/ingestion/base.py` | NEW | `BaseIngestionMethod` protocol |

| `modules/ingestion/methods/__init__.py` | NEW | Package init |

| `modules/ingestion/methods/vector.py` | NEW | `VectorIngestion` (extracted from service) |

| `modules/ingestion/methods/document.py` | NEW | `DocumentIngestion` |

| `modules/ingestion/methods/graph.py` | NEW | `GraphIngestion` |

| `modules/ingestion/methods/tree.py` | NEW | `TreeIngestion` |

| `modules/ingestion/chunk/service.py` | REFACTOR | `ingestion_methods: list[BaseIngestionMethod]` |

| `server.py` | REFACTOR | Config fields optional, dynamic assembly, pass `bm25_enabled` to `VectorRetrieval` |



Update imports in:

- `server.py` (remove `KeywordSearch` import, remove `self._keyword_search` field, pass `bm25_*` to `VectorRetrieval`)

- `retrieval/__init__.py` (re-exports if any)

- Tests (no longer directly create `KeywordSearch` or `RetrievalService` with it)



---



## Verification



1. `poe test` — all 487 existing tests pass

2. `poe typecheck` — mypy passes

3. `poe check` — ruff lint passes

4. New unit tests (in `src/rfnry_rag/retrieval/tests/`):

   - `test_document_search.py` — `DocumentRetrieval` with mocked `BaseDocumentStore`

   - `test_graph_search.py` — `GraphRetrieval` with mocked `BaseGraphStore`

   - `test_vector_ingestion.py` — `VectorIngestion` with mocked stores

   - `test_document_ingestion.py` — `DocumentIngestion` with mocked store

   - `test_retrieval_service_methods.py` — `RetrievalService` with method list

   - `test_ingestion_service_methods.py` — `IngestionService` with method list

5. Integration: server init with document-store-only config (no vector DB)

6. Integration: server init with full-stack config (backward compat)



---



## Implementation Order



1. Create `BaseRetrievalMethod` protocol

2. Refactor `VectorRetrieval`: absorb BM25 logic from `KeywordSearch`, add internal RRF fusion

3. Implement `DocumentRetrieval`, `GraphRetrieval` (extract conversion logic from `RetrievalService`)

4. Refactor `RetrievalService` to use method list dispatch

5. Create `BaseIngestionMethod` protocol

6. Implement `VectorIngestion`, `DocumentIngestion`, `GraphIngestion`, `TreeIngestion`

7. Refactor `IngestionService` to use method list dispatch

8. Refactor `RagServer.initialize()` to assemble both pipelines dynamically

9. Update all imports (`server.py`, `__init__.py`, tests)

10. Write unit tests (search methods, ingestion methods, service dispatch)

11. Write integration tests (document-store-only, hybrid, full-stack backward compat)

12. Run full suite: `poe test && poe typecheck && poe check`