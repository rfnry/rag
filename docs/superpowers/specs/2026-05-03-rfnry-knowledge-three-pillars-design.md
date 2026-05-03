# rfnry-knowledge: three-pillars rename (Semantic / Keyword / Entity)

**Date:** 2026-05-03
**Status:** Approved (user-confirmed Q1–Q7)

## Goal

After the provider-decoupling refactor, the library is no longer RAG-shaped — RAG is one approach among many. Reorganize the public surface around three peer retrieval pillars (**Semantic**, **Keyword**, **Entity**) plus the existing `DIRECT` (full-context) mode, with names and folders that reflect that mental model.

## Non-goals

- Behavior changes to fusion, reranking, grounding, telemetry, observability, or BAML routing.
- Backwards-compatibility shims (no aliases — old names just disappear).
- New features beyond the rename + the BM25/FTS fold.

## Decisions

| # | Decision |
|---|----------|
| Q1 | Split BM25 out of `VectorRetrieval` → standalone `KeywordRetrieval`. |
| Q2 | Fold `DocumentRetrieval` (Postgres FTS) into `KeywordRetrieval` as a backend choice. One config field selects backend (`bm25` or `postgres_fts`). |
| Q3 | Drop `StructuredRetrieval` (and the `enrich/` directory). Cross-reference logic that's still wanted moves into `EntityRetrieval`. |
| Q4 | Rename `QueryMode.INDEXED` → `QueryMode.RETRIEVAL`; `QueryMode.FULL_CONTEXT` → `QueryMode.DIRECT`. `AUTO` unchanged. |
| Q5 | Rename ingestion methods: `VectorIngestion` → `SemanticIngestion`, `DocumentIngestion` → `KeywordIngestion`, `GraphIngestion` → `EntityIngestion`. `AnalyzedIngestion` and `DrawingIngestion` stay (they're phased multi-method orchestrators). |
| Q6 | Folder rename: `retrieval/methods/{vector,document,graph}.py` → `retrieval/methods/{semantic,keyword,entity}.py`; same for `ingestion/methods/`. Drop `retrieval/methods/enrich.py` and `retrieval/enrich/`. |
| Q7 | No backwards compat — old names removed entirely; minor 0.x breaking version bump. |

## Final public surface

### Retrieval pillars (all run in parallel inside `RETRIEVAL` mode, fused via RRF)

| Pillar | Class | What it does | Backend(s) |
|---|---|---|---|
| Semantic | `SemanticRetrieval` | dense embeddings + (optional) sparse hybrid via `BaseSparseEmbeddings` | vector store (Qdrant) |
| Keyword | `KeywordRetrieval` | exact-token / lexical match | `backend="bm25"` (in-memory over vector store payloads) OR `backend="postgres_fts"` (Postgres FTS + substring on the document store) |
| Entity | `EntityRetrieval` | entity lookup + N-hop graph traversal | graph store (Neo4j) |

### Ingestion pillars

| Pillar | Class | Writes |
|---|---|---|
| Semantic | `SemanticIngestion` | embeddings to vector store |
| Keyword | `KeywordIngestion` | text content to document store (powers Postgres FTS); BM25 path needs no separate ingestion (reads vector store payloads at query time) |
| Entity | `EntityIngestion` | entities + relations to graph store |

`AnalyzedIngestion` and `DrawingIngestion` are kept as phased multi-method orchestrators; their internal references to old names get updated.

### `QueryMode`

```python
class QueryMode(Enum):
    RETRIEVAL = "retrieval"   # was INDEXED — run the three pillars in parallel
    DIRECT    = "direct"      # was FULL_CONTEXT — load corpus into prompt, skip retrieval
    AUTO      = "auto"        # corpus-token threshold dispatches between the two
```

Telemetry row `mode` field strings update accordingly: `"retrieval"` and `"direct"`.

## Code shape

### `KeywordRetrieval` (new)

```python
@dataclass
class KeywordRetrieval(BaseRetrievalMethod):
    name: str = "keyword"
    weight: float = 1.0
    top_k: int | None = None
    backend: Literal["bm25", "postgres_fts"] = "bm25"

    # backend="bm25" requires:
    vector_store: BaseVectorStore | None = None
    bm25_max_chunks: int = 200_000
    bm25_max_indexes: int = 16
    parent_expansion: bool = False

    # backend="postgres_fts" requires:
    document_store: BaseDocumentStore | None = None
    use_substring_fallback: bool = True

    def __post_init__(self):
        if self.backend == "bm25" and self.vector_store is None:
            raise ConfigurationError(
                "KeywordRetrieval(backend='bm25') requires vector_store"
            )
        if self.backend == "postgres_fts" and self.document_store is None:
            raise ConfigurationError(
                "KeywordRetrieval(backend='postgres_fts') requires document_store"
            )

    async def search(self, query, top_k, filters=None, knowledge_id=None) -> list[RetrievedChunk]:
        if self.backend == "bm25":
            return await self._bm25_search(...)
        return await self._postgres_fts_search(...)
```

The internals lift the existing `_keyword_search` / `_bm25_*` code from `VectorRetrieval` and the existing `DocumentRetrieval` body, unchanged. No behavior change beyond the new dispatch.

### `SemanticRetrieval` (renamed)

`VectorRetrieval` minus the BM25 paths. Hybrid dense+sparse stays (it's still semantic). The `bm25_enabled` / `bm25_max_chunks` / `bm25_max_indexes` / `parent_expansion`-when-keyword fields move to `KeywordRetrieval`. `parent_expansion` for the dense path stays here.

### `EntityRetrieval` (renamed)

`GraphRetrieval` body unchanged.

### `RetrievalConfig`

```python
@dataclass
class RetrievalConfig:
    methods: list[BaseRetrievalMethod] = field(default_factory=list)
    top_k: int = 5
    cross_reference_enrichment: bool = True   # used by EntityRetrieval (was StructuredRetrieval)
    reranker: BaseReranking | None = None
    source_type_weights: dict[str, float] | None = None
```

### Engine query plumbing

```python
mode = self._config.routing.mode
row = QueryTelemetryRow(
    ...,
    mode="retrieval" if mode == QueryMode.RETRIEVAL else "direct",
    routing_decision=mode.name.lower(),
    ...
)

if mode == QueryMode.RETRIEVAL:
    chunks, trace = await self._retrieval_service.retrieve(...)
elif mode == QueryMode.DIRECT:
    corpus = await self._load_full_corpus(knowledge_id)
    ...
```

`AUTO` dispatch unchanged in behavior; just maps to the new enum values.

## File layout (after)

```
src/rfnry_knowledge/
├── retrieval/
│   ├── base.py
│   ├── namespace.py
│   ├── methods/
│   │   ├── __init__.py
│   │   ├── semantic.py        # was vector.py (without BM25)
│   │   ├── keyword.py         # NEW (BM25 from vector.py + FTS from document.py)
│   │   └── entity.py          # was graph.py (with cross-ref logic from enrich/)
│   └── search/
│       ├── service.py
│       ├── fusion.py
│       ├── keyword.py         # BM25 helper (kept; used internally by methods/keyword.py)
│       └── reranking/
├── ingestion/
│   ├── methods/
│   │   ├── semantic.py        # was vector.py
│   │   ├── keyword.py         # was document.py
│   │   ├── entity.py          # was graph.py
│   │   ├── analyzed.py        # stays
│   │   └── drawing.py         # stays
└── ... (everything else unchanged)
```

Removed:
- `retrieval/methods/enrich.py` (StructuredRetrieval class)
- `retrieval/enrich/` directory (StructuredRetrievalService, helpers folded into EntityRetrieval where still needed)
- `retrieval/methods/document.py`, `retrieval/methods/vector.py`, `retrieval/methods/graph.py` (renamed)
- `ingestion/methods/document.py`, `ingestion/methods/vector.py`, `ingestion/methods/graph.py` (renamed)

## Breaking changes (no shims)

- `from rfnry_knowledge import VectorRetrieval` → `SemanticRetrieval`
- `from rfnry_knowledge import DocumentRetrieval` → `KeywordRetrieval(backend="postgres_fts", document_store=...)`
- BM25 was: `VectorRetrieval(..., bm25_enabled=True)` → now: separate `KeywordRetrieval(backend="bm25", vector_store=...)` in the methods list.
- `from rfnry_knowledge import GraphRetrieval` → `EntityRetrieval`
- `from rfnry_knowledge import StructuredRetrieval` → deleted; the cross-reference behavior lives in `EntityRetrieval`.
- `from rfnry_knowledge import VectorIngestion` → `SemanticIngestion`
- `from rfnry_knowledge import DocumentIngestion` → `KeywordIngestion`
- `from rfnry_knowledge import GraphIngestion` → `EntityIngestion`
- `QueryMode.INDEXED` → `QueryMode.RETRIEVAL`
- `QueryMode.FULL_CONTEXT` → `QueryMode.DIRECT`
- Telemetry `QueryTelemetryRow.mode` strings: `"indexed"` → `"retrieval"`, `"full_context"` → `"direct"`.

## Validation

- `poe typecheck` clean.
- `poe check` + `poe format` clean.
- `poe test` green.
- Grep audit: no remaining references to `VectorRetrieval`, `DocumentRetrieval`, `GraphRetrieval`, `StructuredRetrieval`, `VectorIngestion`, `DocumentIngestion`, `GraphIngestion`, `QueryMode.INDEXED`, `QueryMode.FULL_CONTEXT`, or the deleted `enrich/` paths anywhere in `src/` or `tests/`.

## Execution

Single sweep on `main`. Commit message captures the breaking-change scope. Push to remote.
