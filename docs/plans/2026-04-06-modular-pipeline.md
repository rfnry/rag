# Plan: Modular Retrieval & Ingestion Pipeline


## Context

x64rag hard-codes search paths (vector, BM25, document FTS, graph) and ingestion paths (embeddings, document storage, graph extraction, tree indexing) directly into `RetrievalService` and `IngestionService`. Adding a new method means touching the orchestrator. Using document-level search without embeddings is impossible because `vector_store` and `embeddings` are required fields.

**Goal:** Formalize search and ingestion as protocol-based plugin architectures. The server assembles the active pipeline from what's configured. No mandatory vector DB or embeddings -- at least one retrieval path must be activatable. Methods are self-contained, carry their own configuration, handle their own errors, and are exposed to users via a clean namespace API.


## Architecture

```
RagServer
  |
  |-- _validate_config()          Fail-fast cross-config checks
  |
  |-- initialize()                Assembles methods from config
  |     |
  |     |-- ingestion methods:    [VectorIngestion, DocumentIngestion, GraphIngestion, TreeIngestion]
  |     |-- retrieval methods:    [VectorRetrieval, DocumentRetrieval, GraphRetrieval]
  |     |
  |     |-- MethodNamespace       rag.retrieval.vector / rag.ingestion.document
  |     |-- IngestionService      receives list[BaseIngestionMethod]
  |     |-- RetrievalService      receives list[BaseRetrievalMethod]
  |
  |-- rag.retrieval               MethodNamespace[BaseRetrievalMethod]  (public)
  |-- rag.ingestion               MethodNamespace[BaseIngestionMethod]  (public)
```

### Method Naming

Methods are named by what they operate on, not the paradigm:

- `VectorRetrieval` -- dense + SPLADE + BM25 on chunks (internal RRF fusion)
- `DocumentRetrieval` -- FTS/BM25 on full documents
- `GraphRetrieval` -- entity lookup + N-hop traversal
- `VectorIngestion`, `DocumentIngestion`, `GraphIngestion`, `TreeIngestion` -- same convention

### VectorRetrieval Internal Fusion

`VectorRetrieval` is not a single strategy -- it's a family of chunk-level strategies:
- Dense vector search (always)
- Hybrid dense + SPLADE (if `sparse_embeddings` configured)
- BM25 on chunks (if `bm25_enabled` and not `sparse_embeddings`)

It runs these internally, fuses via RRF, returns one `list[RetrievedChunk]`. BM25 cache, invalidation, and LRU eviction move from `KeywordSearch` into `VectorRetrieval` as private internals.


## MethodNamespace

Generic container that exposes methods as attributes and supports iteration.

**File:** `src/rfnry_rag/retrieval/modules/namespace.py`

```python
class MethodNamespace[T]:
    """Exposes pipeline methods as attributes and supports iteration."""

    def __init__(self, methods: list[T]) -> None:
        self._methods: dict[str, T] = {}
        for method in methods:
            self._methods[method.name] = method

    def __getattr__(self, name: str) -> T:
        try:
            return self._methods[name]
        except KeyError:
            raise AttributeError(f"No method '{name}' configured") from None

    def __iter__(self):
        return iter(self._methods.values())

    def __len__(self) -> int:
        return len(self._methods)

    def __contains__(self, name: str) -> bool:
        return name in self._methods
```

**Usage:**

```python
# Attribute access
chunks = await rag.retrieval.vector.search("query", top_k=20)

# Iteration
for method in rag.retrieval:
    results = await method.search("query", top_k=10)

# Check availability
if "graph" in rag.retrieval:
    ...
```


## Retrieval Protocol & Methods

### BaseRetrievalMethod Protocol

**File:** `src/rfnry_rag/retrieval/modules/retrieval/base.py`

```python
class BaseRetrievalMethod(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def weight(self) -> float: ...

    async def search(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
        knowledge_id: str | None = None,
    ) -> list[RetrievedChunk]: ...
```

### VectorRetrieval (refactored from VectorSearch, absorbs KeywordSearch)

**File:** `src/rfnry_rag/retrieval/modules/retrieval/methods/vector.py`

Renamed from `VectorSearch`. Absorbs BM25 logic from `KeywordSearch` internally. Constructor takes `bm25_enabled` and `bm25_max_indexes`. Adds `name` and `weight` properties. Aligns `search()` signature with protocol.

### DocumentRetrieval (new)

**File:** `src/rfnry_rag/retrieval/modules/retrieval/methods/document.py`

Wraps `BaseDocumentStore.search_content()`. Converts `ContentMatch` to `RetrievedChunk`. Logic extracted from `RetrievalService._content_matches_to_chunks()`.

### GraphRetrieval (new)

**File:** `src/rfnry_rag/retrieval/modules/retrieval/methods/graph.py`

Wraps `BaseGraphStore.query_graph()`. Converts `GraphResult` to `RetrievedChunk`. Logic extracted from `RetrievalService._graph_results_to_chunks()`.


## Ingestion Protocol & Methods

### BaseIngestionMethod Protocol

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

### VectorIngestion (new, extracted from IngestionService)

**File:** `src/rfnry_rag/retrieval/modules/ingestion/methods/vector.py`

Extracted from `IngestionService._embed_and_store_incremental()` and `_build_points()`. Handles embed, build VectorPoint, upsert. Owns `_embed_sparse_safe()`.

### DocumentIngestion (new)

**File:** `src/rfnry_rag/retrieval/modules/ingestion/methods/document.py`

Wraps `BaseDocumentStore.store_content()` / `delete_content()`.

### GraphIngestion (new)

**File:** `src/rfnry_rag/retrieval/modules/ingestion/methods/graph.py`

Wraps graph entity extraction + `BaseGraphStore`.

### TreeIngestion (new)

**File:** `src/rfnry_rag/retrieval/modules/ingestion/methods/tree.py`

Thin wrapper around existing `TreeIndexingService`. Only runs when `pages` are provided.


## Per-Method Error Isolation

Each method handles its own errors. The service never catches method exceptions -- because methods never raise them. A failed method returns an empty list and logs a warning with timing.

```python
# Every concrete retrieval method follows this pattern
async def search(self, query: str, top_k: int, **kwargs) -> list[RetrievedChunk]:
    start = time.perf_counter()
    try:
        results = await self._store.query(...)
        elapsed = (time.perf_counter() - start) * 1000
        logger.info("%d results in %.1fms", len(results), elapsed)
        return self._convert(results)
    except Exception as exc:
        elapsed = (time.perf_counter() - start) * 1000
        logger.warning("failed in %.1fms — %s", elapsed, exc)
        return []
```

Same pattern for `BaseIngestionMethod.ingest()` -- catch, log, continue. If `GraphIngestion` fails, vector and document ingestion still complete.

### Logging Convention

Each method file uses a hierarchical logger name:

```python
# retrieval/methods/vector.py
logger = get_logger("retrieval.methods.vector")

# ingestion/methods/document.py
logger = get_logger("ingestion.methods.document")
```

Output:
```
[retrieval.methods.vector] 42 results in 12.3ms
[retrieval.methods.graph] failed in 5012.1ms — connection timeout
[ingestion.methods.vector] 156 chunks embedded in 891.2ms
[ingestion.methods.tree] skipped — no pages provided
```


## RetrievalService Refactor

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

`_search_single_query()` becomes a simple parallel dispatch:

```python
gathered = await asyncio.gather(*(
    method.search(query=query, top_k=fetch_k, filters=filters, knowledge_id=knowledge_id)
    for method in self._retrieval_methods
))
result_lists = [results for results in gathered if results]
```

No try/except -- methods handle their own errors. Remove `_content_matches_to_chunks()` and `_graph_results_to_chunks()` (moved to method classes). Remove `_keyword_search` reference.

Keep `_apply_source_weights()` and `_build_filters()`. Tree chunks still injected pre-RRF via `tree_chunks` parameter on `retrieve()`.


## IngestionService Refactor

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
1. Parse file, chunk pages, build full_text
2. For each method: `await method.ingest(source_id, ..., full_text, chunks, pages, ...)`
3. Create source in metadata_store (if present)
4. Fire `on_ingestion_complete` callback

`_check_duplicate()`, `_resolve_weight()` stay on `IngestionService` (orchestration concerns).


## Centralized Config Validation

**In:** `src/rfnry_rag/retrieval/server.py` -- new `_validate_config()` method on `RagServer`.

```python
async def initialize(self) -> None:
    self._validate_config()
    # ... then wire everything

def _validate_config(self) -> None:
    cfg = self._config
    p = cfg.persistence
    i = cfg.ingestion

    has_vector = p.vector_store is not None and i.embeddings is not None
    has_document = p.document_store is not None
    has_graph = p.graph_store is not None

    if not any([has_vector, has_document, has_graph]):
        raise ConfigurationError(
            "At least one retrieval path must be configured: "
            "vector (vector_store + embeddings), "
            "document (document_store), or graph (graph_store)"
        )

    if p.vector_store and not i.embeddings:
        raise ConfigurationError("vector_store requires embeddings")
    if i.embeddings and not p.vector_store:
        raise ConfigurationError("embeddings requires vector_store")

    if has_graph and not i.lm_config:
        raise ConfigurationError(
            "graph_store requires ingestion.lm_config for entity extraction"
        )

    if cfg.tree_indexing.enabled and not p.metadata_store:
        raise ConfigurationError("tree_indexing requires metadata_store")
    if cfg.tree_search.enabled and not p.metadata_store:
        raise ConfigurationError("tree_search requires metadata_store")
```

### Config Changes

- `PersistenceConfig.vector_store` becomes `BaseVectorStore | None = None` (was required)
- `IngestionConfig.embeddings` becomes `BaseEmbeddings | None = None` (was required)
- Per-field validation (`chunk_size > 0`, `top_k > 0`) stays in `__post_init__`


## Server Assembly

**File:** `src/rfnry_rag/retrieval/server.py` -- inside `initialize()`, after `_validate_config()`:

```python
ingestion_methods: list[BaseIngestionMethod] = []
retrieval_methods: list[BaseRetrievalMethod] = []

# Vector path
if p.vector_store and i.embeddings:
    vector_size = await i.embeddings.embedding_dimension()
    await p.vector_store.initialize(vector_size)

    ingestion_methods.append(VectorIngestion(
        vector_store=p.vector_store,
        embeddings=i.embeddings,
        sparse_embeddings=i.sparse_embeddings,
    ))
    retrieval_methods.append(VectorRetrieval(
        vector_store=p.vector_store,
        embeddings=i.embeddings,
        sparse_embeddings=i.sparse_embeddings,
        bm25_enabled=r.bm25_enabled and not i.sparse_embeddings,
        bm25_max_indexes=r.bm25_max_indexes,
        weight=1.0,
    ))

# Document path
if p.document_store:
    ingestion_methods.append(DocumentIngestion(document_store=p.document_store))
    retrieval_methods.append(DocumentRetrieval(document_store=p.document_store, weight=0.8))

# Graph path
if p.graph_store:
    ingestion_methods.append(GraphIngestion(graph_store=p.graph_store, lm_config=i.lm_config))
    retrieval_methods.append(GraphRetrieval(graph_store=p.graph_store, weight=0.7))

# Tree path (ingestion only)
if cfg.tree_indexing.enabled and p.metadata_store:
    ingestion_methods.append(TreeIngestion(tree_service=self._tree_indexing_service))

# Build namespaces (public API)
self._retrieval_namespace = MethodNamespace(retrieval_methods)
self._ingestion_namespace = MethodNamespace(ingestion_methods)

# Build services (internal wiring)
self._retrieval_service = RetrievalService(
    retrieval_methods=retrieval_methods,
    reranking=r.reranker,
    top_k=r.top_k,
    source_type_weights=r.source_type_weights,
    query_rewriter=r.query_rewriter,
    chunk_refiner=r.chunk_refiner,
)
```

### Backward Compatibility

- Existing configs with `vector_store` + `embeddings` work identically
- `StructuredIngestionService` unchanged (still takes stores directly)
- `StructuredRetrievalService` unchanged (vector-only)
- Collection-scoped retrieval pipelines rebuilt with same pattern


---


## Migration Overview

This section maps every breaking change to the files that consume the old API. Internal services are not part of the public SDK (`__init__.py` only exports `RagServer` and config classes), so most migration is internal to the package.

### Public API Changes

**`PersistenceConfig`** -- `vector_store` becomes optional (`BaseVectorStore | None = None`).

Consumers:
- `src/rfnry_rag/retrieval/cli/config.py` -- builds `PersistenceConfig` from TOML. No change needed (still passes a vector store when configured).
- User code constructing `PersistenceConfig` directly -- now valid to omit `vector_store` if `document_store` or `graph_store` is provided.

**`IngestionConfig`** -- `embeddings` becomes optional (`BaseEmbeddings | None = None`).

Consumers:
- `src/rfnry_rag/retrieval/cli/config.py` -- same as above.
- User code constructing `IngestionConfig` directly -- now valid to omit `embeddings` if not using vector path.

**`RagServer` new properties** -- `rag.retrieval` and `rag.ingestion` are new. No existing code breaks.

**New public exports** -- Add to `retrieval/__init__.py`:
- `BaseRetrievalMethod`, `BaseIngestionMethod` (protocols)
- `VectorRetrieval`, `DocumentRetrieval`, `GraphRetrieval` (methods)
- `VectorIngestion`, `DocumentIngestion`, `GraphIngestion`, `TreeIngestion` (methods)
- `MethodNamespace` (generic container)

### Internal: server.py

**Removed fields:**
- `self._keyword_search` -- BM25 logic moves into `VectorRetrieval`
- `self._unstructured_retrieval` -- replaced by `self._retrieval_service` + `self._retrieval_namespace`
- `self._unstructured_ingestion` -- replaced by `self._ingestion_service` + `self._ingestion_namespace`

**Removed imports:**
- `KeywordSearch` from `modules.retrieval.search.keyword`
- `VectorSearch` from `modules.retrieval.search.vector`

**New imports:**
- `VectorRetrieval`, `DocumentRetrieval`, `GraphRetrieval` from `modules.retrieval.methods.*`
- `VectorIngestion`, `DocumentIngestion`, `GraphIngestion`, `TreeIngestion` from `modules.ingestion.methods.*`
- `MethodNamespace` from `modules.namespace`
- `BaseRetrievalMethod` from `modules.retrieval.base`
- `BaseIngestionMethod` from `modules.ingestion.base`

**Callback changes:**
- `_on_ingestion_complete` currently calls `self._keyword_search.invalidate()`. After migration, BM25 invalidation is internal to `VectorRetrieval`. The callback should notify retrieval methods that need cache invalidation. Options:
  - Add `on_ingestion_complete(knowledge_id)` to `BaseRetrievalMethod` (optional, default no-op)
  - Or have `VectorRetrieval` expose an `invalidate_cache(knowledge_id)` method called by the server directly
- `_on_source_removed` -- same pattern.

**`_build_retrieval_pipeline`** (per-collection pipelines):
- Currently constructs `VectorSearch` + `KeywordSearch` + `RetrievalService` per collection.
- After migration: constructs method instances per scoped store, passes as list to `RetrievalService`.

**`_get_retrieval()` / `_get_ingestion()`:**
- Currently return `self._unstructured_retrieval` / `self._unstructured_ingestion`.
- After migration: return `self._retrieval_service` / `self._ingestion_service`.

### Internal: RetrievalService

**File:** `src/rfnry_rag/retrieval/modules/retrieval/search/service.py`

**Constructor changes:**
```
# Before
vector_search: VectorSearch
keyword_search: KeywordSearch | None
document_store: BaseDocumentStore | None
graph_store: BaseGraphStore | None

# After
retrieval_methods: list[BaseRetrievalMethod]
```

**Removed methods:**
- `_content_matches_to_chunks()` -- moves to `DocumentRetrieval`
- `_graph_results_to_chunks()` -- moves to `GraphRetrieval`

**Removed references:**
- `self._keyword_search`
- `self._vector_search`
- `self._document_store`
- `self._graph_store`

### Internal: IngestionService

**File:** `src/rfnry_rag/retrieval/modules/ingestion/chunk/service.py`

**Constructor changes:**
```
# Before
embeddings: BaseEmbeddings
vector_store: BaseVectorStore
document_store: BaseDocumentStore | None
sparse_embeddings: BaseSparseEmbeddings | None

# After
ingestion_methods: list[BaseIngestionMethod]
```

**Extracted methods:**
- `_embed_and_store_incremental()` -- moves to `VectorIngestion`
- `_build_points()` -- moves to `VectorIngestion`
- `_embed_sparse_safe()` -- moves to `VectorIngestion`
- Document store blocks (`if self._document_store: ...`) -- moves to `DocumentIngestion`

### Internal: VectorSearch -> VectorRetrieval

**File rename:** `modules/retrieval/search/vector.py` -> `modules/retrieval/methods/vector.py`
**Class rename:** `VectorSearch` -> `VectorRetrieval`

**Additions:**
- `name` property (returns `"vector"`)
- `weight` property (constructor param)
- BM25 logic from `KeywordSearch` (cache, invalidation, LRU eviction)
- `bm25_enabled` and `bm25_max_indexes` constructor params
- Internal RRF fusion of dense + sparse/BM25 before returning results

**Signature change:** `search()` aligns with `BaseRetrievalMethod` protocol (adds `knowledge_id` param).

### Internal: KeywordSearch

**File:** `modules/retrieval/search/keyword.py` -- deprecated. Keep for backward compatibility if directly imported, but no longer wired by server or tests.

### Tests

All test files that construct services directly need updates:

**`tests/test_fulltext_retrieval.py`:**
- Currently constructs `RetrievalService` with mocked `VectorSearch` and `document_store`
- After: construct `RetrievalService` with `[mock_vector_method, mock_document_method]`
- Tests for `_content_matches_to_chunks()` move to `tests/test_document_retrieval.py`

**`tests/test_graph_retrieval.py`:**
- Currently constructs `RetrievalService` with mocked `VectorSearch` and `graph_store`
- After: construct `RetrievalService` with `[mock_vector_method, mock_graph_method]`
- Tests for `_graph_results_to_chunks()` move to `tests/test_graph_retrieval.py` (testing `GraphRetrieval` directly)

**`tests/test_hybrid_retrieval.py`:**
- Currently constructs `VectorSearch` directly
- After: construct `VectorRetrieval` with same mocks, add `weight` param

**`tests/test_query_rewriting.py`:**
- Currently constructs `RetrievalService` with mocked `VectorSearch`
- After: construct `RetrievalService` with `[mock_vector_method]`

**`tests/test_ingestion_advanced.py`:**
- Currently constructs `IngestionService` with mocked embeddings, vector_store
- After: construct `IngestionService` with `[mock_vector_ingestion]`

**`tests/test_server_query.py`:**
- Currently mocks `_unstructured_retrieval` and `_generation_service`
- After: mock `_retrieval_service` and `_generation_service`

### New Test Files

- `tests/test_document_retrieval.py` -- `DocumentRetrieval` with mocked `BaseDocumentStore`
- `tests/test_vector_ingestion.py` -- `VectorIngestion` with mocked stores
- `tests/test_document_ingestion.py` -- `DocumentIngestion` with mocked store
- `tests/test_retrieval_service_methods.py` -- `RetrievalService` with method list dispatch
- `tests/test_ingestion_service_methods.py` -- `IngestionService` with method list dispatch
- `tests/test_namespace.py` -- `MethodNamespace` attribute access, iteration, containment


---


## Critical Files

| File | Action | Description |
|------|--------|-------------|
| `modules/namespace.py` | NEW | `MethodNamespace[T]` generic container |
| `modules/retrieval/base.py` | NEW | `BaseRetrievalMethod` protocol |
| `modules/retrieval/methods/__init__.py` | NEW | Package init |
| `modules/retrieval/methods/vector.py` | NEW | `VectorRetrieval` (from `VectorSearch` + `KeywordSearch` BM25) |
| `modules/retrieval/methods/document.py` | NEW | `DocumentRetrieval` |
| `modules/retrieval/methods/graph.py` | NEW | `GraphRetrieval` |
| `modules/retrieval/search/vector.py` | DELETE | Old `VectorSearch` -- moved to methods/ |
| `modules/retrieval/search/keyword.py` | DEPRECATE | Keep file, no longer wired |
| `modules/retrieval/search/service.py` | REFACTOR | Method list dispatch, remove hardcoded branches |
| `modules/ingestion/base.py` | NEW | `BaseIngestionMethod` protocol |
| `modules/ingestion/methods/__init__.py` | NEW | Package init |
| `modules/ingestion/methods/vector.py` | NEW | `VectorIngestion` |
| `modules/ingestion/methods/document.py` | NEW | `DocumentIngestion` |
| `modules/ingestion/methods/graph.py` | NEW | `GraphIngestion` |
| `modules/ingestion/methods/tree.py` | NEW | `TreeIngestion` |
| `modules/ingestion/chunk/service.py` | REFACTOR | Method list dispatch |
| `server.py` | REFACTOR | Optional configs, `_validate_config()`, dynamic assembly, namespaces |
| `__init__.py` | UPDATE | Add new public exports |


## Implementation Order

1. Create `MethodNamespace[T]` generic class
2. Create `BaseRetrievalMethod` protocol
3. Implement `VectorRetrieval` (refactor from `VectorSearch`, absorb BM25 from `KeywordSearch`)
4. Implement `DocumentRetrieval` (extract from `RetrievalService`)
5. Implement `GraphRetrieval` (extract from `RetrievalService`)
6. Refactor `RetrievalService` to method list dispatch
7. Create `BaseIngestionMethod` protocol
8. Implement `VectorIngestion` (extract from `IngestionService`)
9. Implement `DocumentIngestion`, `GraphIngestion`, `TreeIngestion`
10. Refactor `IngestionService` to method list dispatch
11. Add `_validate_config()` to `RagServer`
12. Refactor `RagServer.initialize()` for dynamic assembly + namespaces
13. Make `PersistenceConfig.vector_store` and `IngestionConfig.embeddings` optional
14. Update `__init__.py` with new exports
15. Migrate existing tests to new constructors
16. Write new unit tests (methods, services, namespace)
17. Write integration tests (document-only, hybrid, full-stack backward compat)
18. Run full suite: `poe test && poe typecheck && poe check`


## Verification

1. `poe test` -- all existing tests pass (after migration)
2. `poe typecheck` -- mypy passes
3. `poe check` -- ruff lint passes
4. New unit tests cover all method classes, service dispatch, namespace behavior
5. Integration: server init with document-store-only config (no vector DB)
6. Integration: server init with full-stack config (backward compat)
