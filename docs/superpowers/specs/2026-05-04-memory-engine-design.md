# Memory Engine — Design Spec

**Date:** 2026-05-04
**Status:** Approved (pending user spec review)
**Authors:** brainstorming session, project owner
**Companion to:** `KnowledgeEngine` in `rfnry-knowledge`

---

## 1. Motivation

`rfnry-knowledge` ships retrieval and ingestion engines today: `KnowledgeEngine` over a corpus, organized around three pillars (Semantic, Keyword, Entity). The package was renamed from `rfnry-retrieval` because retrieval is no longer the only thing it does.

This spec adds a second engine — `MemoryEngine` — for **storing and recalling memories produced by an agentic system over time**. The reference implementation is `mem0`, but we ship the algorithm without the bloat: ~200 lines of pipeline plus a lean BAML extraction prompt, reusing the three pillars verbatim. We do not ship a server, plugins, vendor integrations, an evaluation harness, or vendor-specific clients.

### Guiding principle: engine, not framework

> The memory engine stores and retrieves memory. Policy, prompt assembly, knowledge-vs-memory weighting, audit destinations, versioning rules, decay, summarization — all live in the consumer.

This principle resolves every design fork below. When in doubt: ship the substrate, expose the events, let the consumer compose.

### What survives mem0 vs. what we drop

mem0 v3 distilled to its core:

1. Additive LLM extraction of atomic facts from a conversation.
2. Hash dedup against existing memories.
3. Vector store with payload metadata (scope, timestamps, hash, lemmatized text, attribution).
4. A second vector collection of entities (because they ripped out their graph store).
5. Hybrid retrieval: semantic + BM25 + entity boost.
6. SQLite mutation history.
7. Manual `update`/`delete` (no LLM reconciliation, no decay, no TTL).

What we keep, mapped to existing rfnry surface:

| mem0 capability | rfnry mechanism |
|---|---|
| Semantic vector search | **Semantic** pillar (existing) |
| BM25 on lemmatized text | **Keyword** pillar (existing — `bm25` and `postgres_fts` backends) |
| Entity boost via flat-vector hack | **Entity** pillar over `BaseGraphStore` (existing — strictly more powerful: n-hop traversal) |
| RRF fusion + rerank | `RetrievalService` (existing) |
| Provider routing | `ProviderClient` + BAML registry (existing) |
| SQLite mutation history | `Observability` events + `Telemetry` rows (existing — consumer-sinkable) |
| Per-user/agent/run scoping | Single opaque `memory_id` (new — orthogonal to existing `knowledge_id`) |
| Additive extraction prompt | New BAML function `ExtractMemories` (~40 lines, lean) |

What we drop and why:

- **mem0's flat entity-vector regression** — we kept the real graph store.
- **Multiple memory types (semantic/episodic/procedural)** — engine, not framework. Consumers encode this in `interaction_metadata` if needed.
- **Procedural memory branch** — same. A consumer that wants to summarize an agent trace into a memory formats it as an `Interaction`.
- **SQLite mutation log** — events + telemetry rows already provide this substrate.
- **Decay / TTL / auto-merge** — policy belongs in the consumer.
- **Bulk `purge` / batch update** — YAGNI for v1; revisit on demand.
- **Generation step inside memory** — no `MemoryEngine.query()` analog of `KnowledgeEngine.query()`. Consumers compose search results into prompts.
- **Cross-`memory_id` graph traversal** — traversal stays inside a `memory_id`. Shared world graphs are the consumer's job to express via shared `memory_id`s.

---

## 2. Architecture

### 2.1 Module placement

```
packages/python/src/rfnry_knowledge/
├── memory/                     # NEW — parallel to ingestion/, retrieval/, generation/
│   ├── __init__.py
│   ├── engine.py               # MemoryEngine
│   ├── manager.py              # MemoryManager (orchestration; mirrors knowledge/manager.py)
│   ├── service.py              # MemoryIngestionService, MemoryRetrievalService thin wrappers
│   ├── extraction.py           # BaseExtractor protocol, DefaultMemoryExtractor (BAML)
│   └── models.py               # Interaction, InteractionTurn, ExtractedMemory, MemoryRow, MemorySearchResult
├── config/
│   └── memory.py               # NEW — MemoryEngineConfig, MemoryIngestionConfig, MemoryRetrievalConfig
├── baml/baml_src/
│   └── extract_memories.baml   # NEW — ExtractMemories function
└── __init__.py                 # re-export MemoryEngine, MemoryEngineConfig, Interaction, InteractionTurn, …
```

### 2.2 Memory and knowledge are fully orthogonal namespaces

A row is either a knowledge chunk (carries `knowledge_id`, no `memory_id`) or a memory row (carries `memory_id`, no `knowledge_id`). Search results never mix the two engines; the SDK ships no fused-search escape hatch in v1. Consumers who want both compose them client-side — they import both engines, call them independently, and merge results in their own prompt assembly.

### 2.3 Storage namespacing — the only mechanical change to existing code

Each existing store impl gains a **collection-prefix constructor parameter** with a knowledge-side default. No protocol changes; this is impl-level.

| Store | Knobs | Notes |
|---|---|---|
| `QdrantVectorStore` | `collection_name: str` | Memory passes a different collection name |
| `PostgresDocumentStore` | `table_name: str` | Memory passes a different table name |
| `Neo4jGraphStore` | `node_label_prefix: str = ""` | Memory passes `"Memory"` → entity nodes become `:MemoryEntity`; edge types prefixed similarly. Memory rows themselves are **not** graph nodes — symmetric with knowledge, where chunks are not graph nodes either |
| `SQLAlchemyMetadataStore` | unchanged | Telemetry rows already discriminate by row type |

Consumers instantiate each store **twice** when they want both engines, against the same or different physical clusters — the SDK only requires that the namespaces be disjoint within whichever cluster is chosen.

### 2.4 Public surface

Re-exported flat from the package root (mirrors how `KnowledgeEngine` is exported today):

```python
from rfnry_knowledge import (
    KnowledgeEngine, KnowledgeEngineConfig,
    MemoryEngine, MemoryEngineConfig,
    MemoryIngestionConfig, MemoryRetrievalConfig,
    Interaction, InteractionTurn,
    MemoryRow, MemorySearchResult, ExtractedMemory,
    BaseExtractor, DefaultMemoryExtractor,
    MemoryError, MemoryNotFoundError, MemoryExtractionError,
)
```

`MemoryManager`, telemetry row types, BAML generated client — internal.

---

## 3. Data model

All frozen dataclasses, all in `memory/models.py`. No new store row schemas — memory rows are `RetrievedChunk`-shaped records with memory-flavored payload.

### `InteractionTurn`

```python
@dataclass(frozen=True)
class InteractionTurn:
    role: str       # opaque consumer-defined label (e.g., "user"/"assistant", "customer"/"agent", "trace")
    content: str    # plain text; SDK does not parse tool calls, attachments, etc.
```

### `Interaction`

```python
@dataclass(frozen=True)
class Interaction:
    turns: tuple[InteractionTurn, ...]
    occurred_at: datetime | None = None              # temporal anchor for relative time language
    metadata: Mapping[str, Any] = field(default_factory=dict)  # passthrough to every produced MemoryRow
```

### `ExtractedMemory`

```python
@dataclass(frozen=True)
class ExtractedMemory:
    text: str                                         # atomic factual statement
    attributed_to: str | None                         # role from the source turn, or None
    linked_memory_row_ids: tuple[str, ...] = ()       # populated only if dedup_context_top_k > 0
```

### `MemoryRow`

```python
@dataclass(frozen=True)
class MemoryRow:
    memory_row_id: str                                # uuid7, time-ordered
    memory_id: str                                    # consumer's opaque scope
    text: str
    text_hash: str                                    # sha256(text.strip().lower())
    attributed_to: str | None
    linked_memory_row_ids: tuple[str, ...]
    created_at: datetime
    updated_at: datetime
    interaction_metadata: Mapping[str, Any]
```

### `MemorySearchResult`

```python
@dataclass(frozen=True)
class MemorySearchResult:
    row: MemoryRow
    score: float                                      # fused score
    pillar_scores: Mapping[str, float]                # {"semantic": ..., "keyword": ..., "entity": ...}
```

### Deliberate omissions

- No memory `type` enum (no `semantic`/`episodic`/`procedural` distinction).
- No `confidence` / `salience` / `importance` field.

---

## 4. The `add()` pipeline

`MemoryEngine.add(interaction: Interaction, memory_id: str) -> tuple[MemoryRow, ...]`

Eight phases. Phases marked ★ are genuinely new compute; the rest delegate to existing machinery.

### Phase 1 — Validate and normalize

- Reject empty `interaction.turns`.
- Reject empty/blank `memory_id`.
- Default `occurred_at` to `datetime.now(timezone.utc)` if `None`.
- Emit `memory.add.start` with `{ memory_id, turn_count, occurred_at }`.

### Phase 2 — Optional dedup-context retrieval ★

Runs only if `MemoryIngestionConfig.dedup_context_top_k > 0`.

- Build a probe query from the last `dedup_context_recent_turns` turns (default 3).
- Run a scoped semantic search against the memory-side vector store, filtered by `memory_id`, `top_k = dedup_context_top_k`.
- Pass the resulting `MemoryRow`s to the extractor as context.

### Phase 3 — Extraction ★

Call `BaseExtractor.extract(interaction, existing_memories=...) -> tuple[ExtractedMemory, ...]`.

The default impl wraps a BAML function `ExtractMemories`. The prompt is lean (target ~40 lines):

> *Given an interaction (sequence of role-tagged turns) and optionally a small list of existing memories, return a list of atomic factual statements that should be remembered. Each statement is self-contained, attributed to the role that produced or implied it, and grounded against the provided `occurred_at`. If a statement is semantically equivalent to one of the existing memories, omit it. If a new statement is closely related to an existing one, include the existing memory's id in `linked_memory_row_ids`.*

Output schema: `list[ExtractedMemory]`.

If the extractor returns zero memories, emit `memory.add.empty` and return `()` — not an error.

### Phase 4 — Hash dedup (unconditional)

For each `ExtractedMemory`:
- Compute `text_hash = sha256(text.strip().lower())`.
- Look up `text_hash` in the memory-side vector store via a payload-only filter, scoped by `memory_id`.
- If a match exists, drop the extracted memory and emit `memory.add.dedup_hit` with `{ memory_id, existing_memory_row_id }`.

This runs even when Phase 2 is off.

**Backend assumption:** the vector store backend must support payload-only filter queries (no embedding required) over the `text_hash` field. Qdrant (the default) does. If a future backend cannot, `add()` falls back to a single payload-only fetch of the `dedup_context_top_k` results from Phase 2 and compares hashes locally; backends that cannot do *either* skip hash dedup with a `memory.add.hash_dedup_skipped` warning event.

### Phase 5 — Build `MemoryRow` records

For each surviving extracted memory:
- Mint `memory_row_id = uuid7()`.
- Set `created_at = updated_at = now()`.
- Copy `interaction.metadata` into `interaction_metadata`.
- Validate `linked_memory_row_ids`: drop any IDs that don't exist in the memory-side store under this `memory_id`. Emit `memory.add.invalid_link` per drop.

### Phase 6 — Three-pillar dispatch (parallel)

For each `MemoryRow`, run the three pillars concurrently against memory-side stores. Each pillar is configured with its own `required: bool` flag from `MemoryIngestionConfig`. Required-pillar failures abort the row; optional-pillar failures are logged and skipped (existing `IngestionService` pattern).

- **Semantic** → `BaseEmbeddings.embed([row.text])` → `BaseVectorStore.add_chunks(...)` against the memory-side collection. Payload: `memory_row_id`, `memory_id`, `text_hash`, `attributed_to`, `linked_memory_row_ids`, `created_at`, `text_lemmatized`, plus flattened `interaction_metadata`.
- **Keyword** → `bm25` backend lemmatizes and stores on the same vector payload (no separate write); `postgres_fts` writes to the memory-side document store keyed by `memory_row_id`.
- **Entity** → run the existing entity-extraction BAML function on `row.text` → write entity nodes to the memory-side graph namespace (e.g., `:MemoryEntity`), each carrying `memory_row_ids` as a property/relation. Symmetric with how the knowledge-side Entity pillar links entities to chunk IDs — there is no `:Memory` node type. `linked_memory_row_ids` from the extractor are stored only as a payload property on the memory row; consumers who want to follow them resolve them via a vector store lookup by ID. This deliberately does not introduce a `(:Memory)-[:LINKED_TO]->(:Memory)` edge type, keeping the graph model symmetric with knowledge.

### Phase 7 — Telemetry row

Write `MemoryAddTelemetryRow`:

```
memory_id, row_count, dropped_dedup_count, dropped_invalid_link_count,
extraction_duration_ms, semantic_duration_ms, keyword_duration_ms, entity_duration_ms,
total_duration_ms, token_usage
```

### Phase 8 — Return and emit completion

Return persisted `MemoryRow`s. Emit `memory.add.success` with `{ memory_id, row_count, dropped_dedup_count }`.

If every row failed its required pillar, emit `memory.add.error` and raise the appropriate `IngestionError` subtype.

### Deliberate omissions in `add()`

- **No semantic dedup at storage time.** Phase 4 catches exact duplicates only. Close-but-not-identical memories accumulate; consumers prune via `update`/`delete`.
- **No procedural memory branch.** A consumer that wants to summarize an agent trace formats it as an `Interaction` with whatever `role` label fits.

---

## 5. `search()`, `update()`, `delete()`

### 5.1 `search(query, memory_id, top_k=10, filters=None) -> tuple[MemorySearchResult, ...]`

A thin wrapper over the existing `RetrievalService`:

1. Validate non-blank `query` and `memory_id`. Emit `memory.search.start`.
2. Construct three pillar instances (`SemanticRetrieval`, `KeywordRetrieval`, `EntityRetrieval`) bound to memory-side store namespaces. Pillar weights from `MemoryRetrievalConfig` (defaults: semantic 0.5, keyword 0.3, entity 0.2).
3. Delegate to `RetrievalService.search(...)` with `filters = {"memory_id": memory_id, **(filters or {})}`. RRF fusion, optional source-type weighting, optional rerank — all reused.
4. Map each `RetrievedChunk` into a `MemorySearchResult`, capturing per-pillar scores from `RetrievalTrace`.
5. Write `MemorySearchTelemetryRow` (paralleling `QueryTelemetryRow`).
6. Emit `memory.search.success` with `{ memory_id, result_count, top_score }`.

**No `RoutingConfig` for memory.** No `FULL_CONTEXT` mode (doesn't fit the memory shape). No generation step.

### 5.2 `update(memory_row_id, new_text, *, memory_id) -> MemoryRow`

In-place mutation across all three pillars. `memory_id` required (avoids a payload roundtrip).

1. Fetch existing row from the memory-side vector store by `memory_row_id` + `memory_id` filter. Missing → `MemoryNotFoundError`.
2. Capture `before` snapshot.
3. **Compute entity diff.** Read the prior entity set for this `memory_row_id` from the graph store (cheap; no LLM call). Re-extract entities on `new_text` via the existing entity-extraction BAML function. Compute `(added_entities, removed_entities)` from the two sets.
4. Mutate:
   - **Vector store**: re-embed, overwrite in place. Update `text`, `text_hash`, `text_lemmatized`, `updated_at`. Preserve `created_at`, `linked_memory_row_ids`, `interaction_metadata`.
   - **Document store** (FTS backend): overwrite the FTS document.
   - **Graph store**: drop the `memory_row_id` reference from `removed_entities`; add it to `added_entities`. Entities that lose their last memory reference are deleted (existing knowledge-side cleanup pattern). No memory-row node exists, so there's nothing to update at the row level.
5. Build `after` row.
6. Emit `memory.update.success` with `{ memory_id, memory_row_id, before: MemoryRow, after: MemoryRow }`. **This is the audit substrate.**
7. Write `MemoryUpdateTelemetryRow`.
8. Return `after`.

Per-pillar errors are isolated (existing partial-success contract). The consumer sees per-pillar events and decides whether to retry.

### 5.3 `delete(memory_row_id, *, memory_id) -> None`

Symmetric, simpler:

1. Validate existence; missing → `MemoryNotFoundError`.
2. Capture `before`.
3. Drop from all three pillars in parallel:
   - Vector store: delete by `memory_row_id`.
   - Document store: delete the FTS row if present.
   - Graph store: drop the `memory_row_id` reference from every entity that mentions it. Entities that lose their last memory reference are deleted (existing knowledge-side cleanup pattern).
4. Emit `memory.delete.success` with `{ memory_id, memory_row_id, before: MemoryRow }`.
5. Write `MemoryDeleteTelemetryRow`.

Hard delete only. Consumers wanting soft-delete sink the events into their own append-only table and never call `delete`.

### 5.4 Bulk operations

Not shipped in v1.

### 5.5 Errors

Added to the existing hierarchy:

- `MemoryError(KnowledgeEngineError)` — base.
- `MemoryNotFoundError(MemoryError, StoreError)` — `update`/`delete` on missing row.
- `MemoryExtractionError(MemoryError, IngestionError)` — extractor failure.

---

## 6. Configs

Three new frozen dataclasses in `config/memory.py`.

### `MemoryIngestionConfig`

```python
@dataclass(frozen=True)
class MemoryIngestionConfig:
    extractor: BaseExtractor                                # required (consumer constructs DefaultMemoryExtractor with provider, or supplies their own impl)
    embeddings: BaseEmbeddings                              # required
    sparse_embeddings: BaseSparseEmbeddings | None = None
    entity_extraction: EntityIngestionConfig | None = None  # None disables Entity pillar
    semantic_required: bool = True
    keyword_required: bool = False
    entity_required: bool = False
    dedup_context_top_k: int = 0                            # 0 = off
    dedup_context_recent_turns: int = 3
    keyword_backend: Literal["bm25", "postgres_fts"] = "bm25"
    bm25_max_chunks: int = 50_000
```

### `MemoryRetrievalConfig`

```python
@dataclass(frozen=True)
class MemoryRetrievalConfig:
    semantic_weight: float = 0.5
    keyword_weight: float = 0.3
    entity_weight: float = 0.2
    entity_hops: int = 2
    rerank: BaseReranking | None = None
    over_fetch_multiplier: int = 4
```

### `MemoryEngineConfig`

```python
@dataclass(frozen=True)
class MemoryEngineConfig:
    ingestion: MemoryIngestionConfig
    retrieval: MemoryRetrievalConfig
    vector_store: BaseVectorStore                           # memory-namespaced
    document_store: BaseDocumentStore | None = None         # required iff keyword_backend="postgres_fts"
    graph_store: BaseGraphStore | None = None               # required iff entity_extraction is not None
    metadata_store: BaseMetadataStore | None = None
    provider: ProviderClient
    observability: Observability = field(default_factory=Observability)
    telemetry: Telemetry = field(default_factory=Telemetry)
```

### Validation (`__post_init__`)

- `keyword_backend == "postgres_fts"` → `document_store` must be set.
- `entity_extraction is not None` → `graph_store` must be set.
- `dedup_context_top_k >= 0`, `dedup_context_recent_turns >= 1`, `bm25_max_chunks > 0`.
- All retrieval weights non-negative; sum > 0.
- Conflicting combinations raise `ConfigurationError`.

---

## 7. Consumer wiring example

```python
from rfnry_knowledge import (
    KnowledgeEngine, KnowledgeEngineConfig,
    MemoryEngine, MemoryEngineConfig,
    MemoryIngestionConfig, MemoryRetrievalConfig,
    Interaction, InteractionTurn,
    QdrantVectorStore, Neo4jGraphStore, PostgresDocumentStore, SQLAlchemyMetadataStore,
    ProviderClient, IngestionConfig, RetrievalConfig, EntityIngestionConfig,
)
from pydantic import SecretStr
from contextlib import AsyncExitStack

provider = ProviderClient(name="anthropic", model="claude-sonnet-4-6", api_key=SecretStr(...))
embeddings = MyVoyageEmbeddings(...)
metadata = SQLAlchemyMetadataStore(dsn=...)

knowledge_cfg = KnowledgeEngineConfig(
    ingestion=IngestionConfig(...),
    retrieval=RetrievalConfig(...),
    metadata_store=metadata,
)

memory_cfg = MemoryEngineConfig(
    ingestion=MemoryIngestionConfig(
        embeddings=embeddings,
        entity_extraction=EntityIngestionConfig(...),
        dedup_context_top_k=10,
    ),
    retrieval=MemoryRetrievalConfig(),
    vector_store=QdrantVectorStore(url=..., collection_name="memory"),
    graph_store=Neo4jGraphStore(url=..., node_label_prefix="Memory"),
    metadata_store=metadata,
    provider=provider,
)

async with AsyncExitStack() as stack:
    knowledge = await stack.enter_async_context(KnowledgeEngine(knowledge_cfg))
    memory    = await stack.enter_async_context(MemoryEngine(memory_cfg))

    interaction = Interaction(turns=(
        InteractionTurn(role="user", content="I just moved to Lisbon and started using FastAPI."),
        InteractionTurn(role="assistant", content="Welcome! What kind of projects?"),
    ))
    await memory.add(interaction, memory_id="user-7f3a")

    docs   = await knowledge.query("FastAPI dependency injection", knowledge_id="python-docs")
    recall = await memory.search("where do they live?", memory_id="user-7f3a")
    # consumer composes prompt from docs.answer, recall, conversation history
```

A consumer that wants only memory imports `MemoryEngine`, configures one vector store (and optionally one graph store), and never sees `KnowledgeEngine` or any document-corpus surface.

---

## 8. Testing strategy

Same harness as today: pytest with `asyncio_mode="auto"`, contract tests around BAML and config bounds, integration tests gated by env, per-pillar unit tests with fakes.

### Contract tests (always run)

- **`test_extract_memories_baml_contract.py`** — fence + schema contract for `ExtractMemories`. Empty interaction → `[]`. `attributed_to` honored. No invented `linked_memory_row_ids`. Drops links to IDs not in the provided existing-memory list.
- **`test_extract_memories_domain_agnostic.py`** — no hardcoded domain vocabulary in the prompt; quality holds across unrelated domains.
- **`test_memory_config_bounds.py`** — bounds, conflicting-combination rejections, all weights non-negative with sum > 0.
- **`test_memory_protocol_contract.py`** — `BaseExtractor` protocol surface; store impls' memory namespaces preserve the existing protocol contracts.

### Unit tests (fakes for stores, mocked BAML)

- **`test_memory_engine_add.py`** — empty extraction; hash collision; invalid `linked_memory_row_ids`; required-pillar failure; optional-pillar failure; dedup-context probe on/off; `interaction_metadata` propagation.
- **`test_memory_engine_search.py`** — `memory_id` filter forwarded; per-pillar scores propagated; empty result; reranker invoked when configured.
- **`test_memory_engine_update.py`** — `MemoryNotFoundError` on missing row; `before`/`after` event payloads match actual rows; entity diff (added/removed) edits graph references correctly; `linked_memory_row_ids` preserved as a payload property across updates; partial pillar failure does not roll back successful pillars.
- **`test_memory_engine_delete.py`** — `MemoryNotFoundError`; cascading three-pillar delete; `before` event payload matches; entity references to the deleted memory are removed; entities that lose their last reference are themselves deleted.

### Integration tests (real backends, env-gated)

- **`test_memory_integration_qdrant.py`** — round-trip on a real Qdrant collection.
- **`test_memory_integration_neo4j.py`** — entity nodes, `LINKED_TO` edges, n-hop traversal.
- **`test_memory_integration_postgres_fts.py`** — keyword backend `postgres_fts` end-to-end.
- **`test_memory_integration_namespacing.py`** — critical: knowledge and memory engines against the *same* physical Qdrant + Neo4j + Postgres, asserting search isolation, graph isolation, and telemetry coexistence.

### Deliberately not tested

- Extraction-quality benchmarks vs mem0 (lives under `evaluation/`, not CI).
- End-to-end "agent uses memory" scenarios (consumer concern).
- Decay / TTL behavior (not shipped).
- Bulk operations (not shipped).

---

## 9. Out-of-scope (v1)

- Bulk operations (`purge(memory_id)`, batch update/delete).
- Soft-delete semantics inside the engine.
- Auto-merge / auto-decay / TTL.
- Cross-`memory_id` graph traversal.
- Built-in fused search across knowledge + memory.
- A `MemoryEngine.query()` analog of `KnowledgeEngine.query()` (no built-in answer synthesis).
- Multiple memory types (semantic/episodic/procedural enums).
- TypeScript port.
- REST server, plugins, vendor integrations.

Each item above can be revisited when a concrete consumer need surfaces. The v1 surface is intentionally minimal so that the engine stays an engine.

---

## 10. Decisions log (locked during brainstorming)

1. **Module shape**: parallel module `rfnry_knowledge.memory/`, flat exports from package root.
2. **Scope identifier**: single opaque `memory_id` (mirrors `knowledge_id`); SDK does not interpret it.
3. **Knowledge↔memory relationship**: fully orthogonal namespaces, separate collections per backend, no fused search shipped.
4. **Knowledge↔memory composition**: consumer-side only. No `search_with_knowledge`.
5. **Extraction prompt**: lean BAML function (~40 lines), trust the model. No 900-line policy doc port.
6. **Extractor swap**: `BaseExtractor` Protocol with `DefaultMemoryExtractor` as the BAML-backed default.
7. **`add()` input**: `Interaction` value object with `tuple[InteractionTurn, ...]`, opaque `role` labels.
8. **Audit log**: none. Mutations shout out via `Observability` events and `Telemetry` rows.
9. **Dedup-context retrieval**: opt-in via `dedup_context_top_k` (default 0). Hash dedup unconditional.
10. **Entity layer**: real `BaseGraphStore` (Neo4j default), n-hop traversal, `memory_id`-scoped subgraphs. Not a flat-vector regression.
11. **Update/delete semantics**: manual, in-place, hard delete. Events carry full `before`/`after` payloads as the audit substrate.
12. **`search`**: thin wrapper over existing `RetrievalService`. No `RoutingConfig`. No generation step.
13. **API surface**: standalone `MemoryEngine`. No `engine.memory.*` namespacing. No `Composite` wrapper.
