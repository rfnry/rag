# rfnry-rag Roadmap

Design document for the next generation of rfnry-rag features. Each item is research-backed, mapped to concrete implementation targets in the current codebase, and scoped for independent delivery. This document is the guide for turning these 8 priorities into executable work.

For the research analysis behind these decisions, see [rfnry-rag Long-Context Strategy](articles/rfnry-rag-long-context-strategy.md).

---

## Status at a glance

Last updated: 2026-04-29. **Roadmap status: complete (2026-04-29).** All
eight in-scope priorities (R3 + R4 + R8 + R1 + R5 + R6 + R2; R7 dropped)
are landed across Phases 1-3.

### Phase 1 — Foundation (independent, small scope)

| Item | Status | Commit / Notes |
|---|---|---|
| **R3** — Document Expansion at Index Time | ✅ Done | `3eb2545` (2026-04-26). Synthetic queries per chunk via opt-in BAML, default off, consumer provides `lm_client`. |
| **R4** — Chunk Position Optimization | ✅ Done | `0038a88` (2026-04-25). `ChunkOrdering` enum (SCORE_DESCENDING / PRIMACY_RECENCY / SANDWICH) on `GenerationConfig`. |
| **R7** — Local Cross-Encoder Reranking | ⏭ Dropped | Out of focus for now; revisit if there's demand. |

Phase 1 closed.

### Phase 2 — Intelligence (observability + adaptation)

| Item | Status | Notes |
|---|---|---|
| **R8** — Retrieval Evaluation & Diagnostics | ✅ Done | 6 commits ending `6db0724` (2026-04-27). R8.1 `RetrievalTrace` + opt-in `trace=True`; R8.2 heuristic `classify_failure`; R8.3 `RagEngine.benchmark` + `rfnry-rag benchmark` CLI. Test count 1035 → 1060 (+25). |
| **R1** — Context-Aware Routing | ✅ Done | 8 commits ending `75139a6` (2026-04-28). R1.1 token counting + corpus loader; R1.2 `QueryMode` enum + `RoutingConfig` + DIRECT mode; R1.3 HYBRID (SELF-ROUTE) with `CheckAnswerability` BAML; R1.4 AUTO mode (recommended for new users). Test count 1060 → 1091 (+31). |
| **R5** — Adaptive Retrieval Parameters | ✅ Done | 7 commits ending `7530c6a` (2026-04-28). R5.1 query classifier (heuristic + opt-in LLM, `ClassifyQueryComplexity` BAML, `AdaptiveRetrievalConfig`); R5.2 dynamic top_k by complexity + task-aware method-weight multipliers by query_type (4 default profiles); R5.3 confidence-based retry loop (2× top_k → rewriter swap → LC escalation via R1.2's DIRECT). New `routing_decision="retrieval_then_direct"`. Test count 1091 → 1120 (+29). |

### Phase 3 — Advanced (build on Phases 1+2)

| Item | Status | Notes |
|---|---|---|
| **R6** — Multi-Hop Iterative Retrieval | ✅ Done | 6 commits ending `5e8f356` (2026-04-29). R6.1 config + BAML scaffold (`IterativeRetrievalConfig`, `DecomposeQuery` BAML); R6.2 `IterativeRetrievalService` + hop loop + engine arm + `routing_decision="iterative"`; R6.3 post-loop DIRECT escalation, new `routing_decision="iterative_then_direct"`, new termination reasons `"low_confidence_escalated"` / `"low_confidence_no_escalation"`, helper lift to `common/grounding.py`. Test count 1120 → 1147 (+27). |
| **R2** — RAPTOR-Style Summarization Retrieval | ✅ Done | 8 commits ending `95bb413` (2026-04-29). R2.1 config + `SummarizeCluster` BAML + `rag_raptor_trees` schema + `RaptorTreeRegistry`; R2.2 `RaptorTreeBuilder` (cluster → cap → summarize → embed → persist → recurse) + atomic blue/green swap + `set_payload` Protocol promotion + GC; R2.3 `RaptorRetrieval` method + engine wiring (eager registry, lazy builder, soft-skip on missing deps, raise on non-SQLA metadata store) + RRF fusion + drawing-corpus skip. Test count 1147 → 1208 (+61). |

### Roadmap complete

With R2 closed, all eight in-scope priorities (R3 + R4 + R8 + R1 + R5 + R6 + R2; R7 dropped) are landed across Phases 1-3. Future work — beyond the explicit "Out of scope" exclusions section below — belongs in a new design document, not this one.

Phase 2 closed. Phase 3 closed.

### Out of scope (explicitly NOT planned)

More embedding providers · Agent / chain orchestration · Fine-tuned dense retrievers · Advanced chunking strategies · Streaming reranking / streaming fusion. See "What This Roadmap Does NOT Include" at the bottom for rationale.

---

## Conventions from Phases A-E

Phases A-E established patterns that new roadmap items MUST follow. Any design in R1-R8 that conflicts with these is either wrong or needs explicit justification in the per-feature design.

### 1. Consumer-agnostic by default

Phase D removed hardcoded electrical/mechanical patterns from the graph mapper; Phase E deleted `QueryAnalysis.domain_hint` for the same reason. Every new feature that uses an LLM prompt MUST:

- Avoid seeding the LLM with domain-specific examples ("valves, motors, wires, terminals", "480V", "SAE 30"). The contract test `test_baml_prompt_domain_agnostic` scans both `baml_src/ingestion/functions.baml` and `baml_src/retrieval/functions.baml` for 11 bias terms and will fail on regressions.
- When the feature needs a vocabulary (entity types, relationship keywords, symbol libraries, relation-type maps), expose it via a consumer-overridable config. See `GraphIngestionConfig` and `DrawingIngestionConfig` for the pattern: empty defaults, consumer supplies domain knowledge, values are validated against an allowlist where applicable.
- **Intent classifications** (factual / comparative / procedural / entity-lookup) are universal and fine — they describe the shape of a query, not the domain. **Domain classifications** (electrical / legal / medical / academic) are not — they leak domain assumption into the SDK.

### 2. New configs must register with the bounds contract

Every new config dataclass added to `IngestionConfig`, `RetrievalConfig`, or `GenerationConfig` must:

- Register in `_CONFIGS_TO_AUDIT` at `test_config_bounds_contract.py` so every numeric field gets bounds validation enforced at import time.
- Add an entry to the "Config defaults and enforced bounds" section of `CLAUDE.md`.
- Validate every numeric field in `__post_init__` with a bounds check, OR carry a `# unbounded: <reason>` marker on the field line.
- Values that map to an allowlist (e.g., Neo4j `ALLOWED_RELATION_TYPES`) must be validated against that allowlist at construction. See `DrawingIngestionConfig.relation_vocabulary` and `GraphIngestionConfig.unclassified_relation_default` for the pattern.

### 3. BAML prompt conventions

Every new BAML function must:

- Classify each parameter in `USER_CONTROLLED_PARAMS` at `test_baml_prompt_fence_contract.py` (either list user-controlled params or an empty list for purely operator-controlled functions). The contract test fails on any unclassified function.
- Fence every user-controlled string parameter with `======== <TAG> START ========` / `======== <TAG> END ========` and an "untrusted" directive in the prompt body.
- Carry no domain-specific examples in the prompt body (see Convention 1).

### 4. Sibling ingestion services, not fold-ins

`AnalyzedIngestionService` (Phase A/B) and `DrawingIngestionService` (Phase C) are parallel entry points into the ingestion layer. Features that operate on the chunk/embedding level (R2 RAPTOR, R3 document expansion, R4 chunk position) apply to the chunk-based services ONLY. Drawing ingestion emits one vector per component, not per chunk — do not wire chunk-level features into the drawing path.

Use `IngestionConfig.*` nested configs for each service (precedent: `tree_search`, `drawings`, `graph`). One config per service, no cross-service knobs.

### 5. Migration pattern for field renames

When renaming a config field (e.g., `contextual_chunking` → `chunk_context_headers` in Phase A), keep the old name as an optional field with a `__post_init__` `DeprecationWarning` and value forwarding. One release of overlap; then remove the deprecated alias.

### 6. Vector payload tagging

Phase A4 established `vector_role: description | raw_text | table_row`. Phase C10 added `drawing_component`. New retrieval paths that produce vectors (R2 RAPTOR) should add a new `vector_role` value (e.g., `raptor_level_N`) rather than overloading an existing role. Consumers filter by role at query time.

---

## R1. Context-Aware Routing

**Status:** Not started
**Impact:** Critical — existential for rfnry-rag's relevance
**Research:** Anthropic Contextual Retrieval, Google DeepMind SELF-ROUTE, LaRA (ICML 2025), Akita (2026)

### Problem

rfnry-rag always retrieves. Every query goes through chunking, embedding search, fusion, reranking — even when the entire knowledge base is 30 pages. In 2026, a 200K-token corpus costs $0.63 to read on Sonnet 4.6, and $0.10 with prompt caching. For corpora under ~500 pages, Direct Context consistently outperforms retrieval in quality (Anthropic, LaRA) while remaining economically viable.

Without this feature, users with small-to-medium knowledge bases have no reason to use rfnry-rag — they get better answers by dumping everything into the prompt.

### Design

Introduce three query modes in `RagEngine`:

**A. Direct Context Mode**

Load all documents for a `knowledge_id` into the generation prompt. No chunking, no embedding search, no fusion. The LLM reads everything and answers directly.

- Requires a way to load full document content per knowledge scope. The `DocumentStore` already stores full text per source (used by `DocumentRetrieval`). For knowledge bases without a document store, fall back to reading stored chunks from the vector store via `scroll()`.
- Prompt structure must maximize cache hit rate: stable corpus prefix (documents), variable suffix (query + conversation history). This maps directly to how BAML `GenerateAnswer` structures its input — the `context` parameter becomes the full corpus, and the `query` parameter remains the variable part.
- Token counting is needed to validate that the corpus fits within model context limits. Track approximate token count per source at ingestion time (store in `Source.metadata` or a new field on `SourceStats`). Aggregate per `knowledge_id` at query time via `KnowledgeManager`.

**B. Retrieval Mode**

Current behavior. Multi-path search, fusion, reranking, refinement. No changes needed — this is the existing pipeline.

**C. Hybrid Mode (SELF-ROUTE)**

RAG-first with LC fallback. Inspired by Google DeepMind's SELF-ROUTE:

1. Run the normal retrieval pipeline
2. Present retrieved chunks to the LLM with an "answerability" check — can you answer this from the retrieved context?
3. If yes: generate answer from chunks (current flow)
4. If no: escalate to Direct Context mode for this query

SELF-ROUTE achieved near-LC accuracy at 35-61% of token cost. 76.8% of queries were resolved by RAG alone on Gemini-1.5-Pro, meaning only ~23% needed the expensive full-context fallback.

### Configuration

```python
from enum import Enum

class QueryMode(Enum):
    RETRIEVAL = "retrieval"       # current behavior
    DIRECT = "direct"             # full corpus in prompt
    HYBRID = "hybrid"             # RAG-first, LC fallback
    AUTO = "auto"                 # system decides based on corpus size

@dataclass
class RoutingConfig:
    mode: QueryMode = QueryMode.RETRIEVAL
    direct_context_threshold: int = 150_000    # tokens — below this, AUTO uses direct
    hybrid_answerability_model: LanguageModelConfig | None = None  # for SELF-ROUTE check
```

Add `routing: RoutingConfig` to `RagServerConfig`. Default remains `RETRIEVAL` for backward compatibility. `AUTO` mode checks corpus size at query time and routes to `DIRECT` or `RETRIEVAL` based on `direct_context_threshold`.

### Implementation Targets

| What | Where | Detail |
|---|---|---|
| Token counting at ingest | `IngestionService.ingest()` | Use `count_tokens()` from `ingestion/chunk/token_counter.py` (Phase A shipped `tiktoken` + cl100k_base encoder). Sum per source, store in `Source.metadata["estimated_tokens"]` |
| Corpus size query | `KnowledgeManager` | New method: `get_corpus_tokens(knowledge_id) -> int` — sum estimated_tokens across all sources in scope |
| Full corpus loader | `RagEngine` | New method: `_load_full_corpus(knowledge_id) -> str` — load all document content from document store or scroll vector store chunks |
| Direct context generation | `GenerationService` | New method or parameter on `generate()` — pass full corpus as context instead of retrieved chunks |
| Answerability check | New BAML function | `CheckAnswerability(query, context) -> {answerable: bool, reasoning: str}`. Fence `query` + `context` as user-controlled; classify in `USER_CONTROLLED_PARAMS`; keep prompt domain-agnostic (Conventions 1 + 3) |
| Routing logic | `RagEngine.query()` | Before calling `_retrieve_chunks()`, check mode. For AUTO, call `get_corpus_tokens()`. For HYBRID, wrap retrieval + answerability check + optional LC fallback |
| Config wiring | `RagServerConfig` | New `RoutingConfig` dataclass, validated in `_validate_config()`. Register in `_CONFIGS_TO_AUDIT`; bounds-check `direct_context_threshold`; add CLAUDE.md bounds entry (Convention 2) |

### Prompt Caching

Direct Context mode and the LC fallback in Hybrid mode benefit from prompt caching. rfnry-rag doesn't control caching directly — it's a provider-level feature (Anthropic's cache_control, OpenAI's automatic caching). But rfnry-rag controls prompt structure, which determines cache hit rates:

- **Stable prefix:** System prompt + full corpus content. This must be identical across queries for the same knowledge_id.
- **Variable suffix:** User query + conversation history.
- **Cache invalidation:** When a source is added/removed/updated within a knowledge_id, the corpus prefix changes and the cache invalidates. This is correct behavior — the corpus has changed.

BAML's `ClientRegistry` and provider configuration may need to expose cache control hints. This is a BAML integration concern, not a core rfnry-rag concern, but the prompt structure must be designed to enable it.

### Dependencies

- Requires document store OR vector store (for chunk scrolling) to be configured
- Hybrid mode requires an LLM config for the answerability check
- No new external dependencies

---

## R2. RAPTOR-Style Summarization Retrieval

**Status:** Not started
**Impact:** High — nearly doubles retrieval accuracy over chunk-based methods
**Research:** Li et al. 2025 (RAPTOR: 38.5% vs chunk-based 20.4%)

### Problem

rfnry-rag's tree retrieval navigates existing document structure (TOC, headings, sections). This works well for structured documents but fails for unstructured content (transcripts, emails, mixed-format corpora). RAPTOR creates structure from content: recursively cluster chunks by semantic similarity, summarize each cluster, then cluster and summarize again — building a hierarchy of increasingly abstract representations.

Li et al. (2025) showed RAPTOR retrieval achieves 38.5% accuracy — nearly double the 20-21% of standard chunk-based methods (BM25, Contriever, OpenAI embeddings). This is the largest single retrieval quality improvement available.

### Design

> **Scope:** RAPTOR operates on the chunk-based ingestion path (`AnalyzedIngestionService` + `VectorIngestion`). It does NOT apply to `DrawingIngestionService` — drawings emit one vector per component, not per chunk (see Phase C10). Consumers with mixed corpora get RAPTOR on the document sources only; drawing sources are untouched (Convention 4).

Two new modules implementing the existing protocols:

**A. `RaptorIngestion` (implements `BaseIngestionMethod`)**

Runs at index time after standard chunking:

1. **Embed leaf chunks** — use the same embedding provider configured in `IngestionConfig`
2. **Cluster** — group chunks by embedding similarity. Use K-Means or HDBSCAN (both already implemented in the Reasoning SDK's `ClusteringService` at `reasoning/modules/clustering/`)
3. **Summarize** — for each cluster, generate a summary via LLM (BAML function). The summary captures the thematic essence of the cluster
4. **Recurse** — embed the summaries, cluster them, summarize again. Repeat until the tree collapses to a single root or reaches `max_levels`
5. **Store** — embed and index all nodes (leaf chunks + summaries at every level) into the vector store. Tag each with `raptor_level` metadata (0 = leaf, 1 = first summary, etc.)

The vector store already supports arbitrary metadata on points (`VectorPoint.payload`). No schema changes needed — just add `raptor_level` and `raptor_cluster_id` to the payload.

**B. `RaptorRetrieval` (implements `BaseRetrievalMethod`)**

Runs at query time alongside other retrieval methods:

1. **Search across all levels** — query the vector store for the top-k chunks, but include summaries from all RAPTOR levels. Higher-level summaries match broader/more abstract queries; leaf chunks match specific/factual queries.
2. **Level weighting** — optionally weight results by level (summaries may get a boost for broad queries, leaves for specific ones). Or let RRF fusion handle it naturally.
3. **Return `list[RetrievedChunk]`** — standard output, participates in RRF fusion with vector, document, graph methods.

### Configuration

```python
@dataclass
class RaptorConfig:
    enabled: bool = False
    max_levels: int = 3                # max recursion depth
    cluster_algorithm: str = "kmeans"  # "kmeans" or "hdbscan"
    clusters_per_level: int = 10       # k for kmeans
    min_cluster_size: int = 5          # for hdbscan
    summary_model: LanguageModelConfig | None = None  # LLM for summarization
    summary_max_tokens: int = 256      # max tokens per summary
```

Add to `IngestionConfig` as `raptor: RaptorConfig`. When `enabled=True`, `RaptorIngestion` is added to the method list in `RagEngine.initialize()`, and `RaptorRetrieval` is added to retrieval methods.

### Implementation Targets

| What | Where | Detail |
|---|---|---|
| Cluster reuse | Import `ClusteringService` from reasoning SDK | Already implements K-Means + HDBSCAN. May need to expose raw cluster assignments (not just samples) |
| BAML summarization | New BAML function `SummarizeCluster` | Input: list of chunk texts (user-controlled — fence and classify per Convention 3). Output: summary string. Prompt must stay domain-agnostic (Convention 1). Add to `retrieval/baml/baml_src/` |
| RaptorIngestion | `modules/ingestion/methods/raptor.py` | New file implementing `BaseIngestionMethod`. Recursive cluster→summarize→embed→store loop |
| RaptorRetrieval | `modules/retrieval/methods/raptor.py` | New file implementing `BaseRetrievalMethod`. Vector search with raptor_level metadata filter |
| Vector payload extension | No schema change | Add `raptor_level`, `raptor_cluster_id`, `raptor_parent_id` to `VectorPoint.payload` dict. Set `vector_role="raptor_summary"` for summary vectors; leaf chunks keep their existing role (`description`, `raw_text`, etc.) per Convention 6 |
| Config wiring | `IngestionConfig` + `RagEngine.initialize()` | Add `RaptorConfig`, conditionally create RAPTOR methods. Register in `_CONFIGS_TO_AUDIT`; bounds-check `max_levels`, `clusters_per_level`, `min_cluster_size`, `summary_max_tokens`; add CLAUDE.md bounds entry (Convention 2) |

### Relationship to Existing Tree Retrieval

RAPTOR and tree retrieval are complementary, not competing:

- **Tree retrieval** navigates explicit document structure (TOC, sections). Best for well-structured documents (manuals, reports, books with chapters).
- **RAPTOR** creates implicit semantic structure from any content. Best for unstructured or mixed corpora.

Both participate in RRF fusion. Users can enable either, both, or neither.

### Dependencies

- Requires `IngestionConfig.embeddings` (for embedding summaries)
- Requires `PersistenceConfig.vector_store` (for storing summary vectors)
- Requires `RaptorConfig.summary_model` (LLM for summarization)
- Uses `ClusteringService` from reasoning SDK — needs to be importable from retrieval context

---

## R3. Document Expansion at Index Time

**Status:** Not started
**Impact:** High — bridges the query-document vocabulary gap
**Research:** BEIR (docT5query outperforms BM25 on 11/18 datasets), Anthropic Contextual Retrieval

### Problem

Users ask questions differently from how documents describe answers. "How do I change the oil?" won't BM25-match a document titled "Lubricant Replacement Procedure." Dense embeddings partially bridge this gap, but BEIR showed that document expansion — generating synthetic queries at index time — outperforms BM25 on 11/18 heterogeneous datasets while maintaining BM25's generalization strength.

rfnry-rag already has `chunk_context_headers` (Phase A; pure string templating that prepends a source/type header to each chunk — *not* LLM-generated; the name was corrected from the misleading `contextual_chunking` in Phase A). Neither does any existing feature generate **questions the chunk answers** — which is a different signal that directly bridges the user-query-to-document-content vocabulary gap. Document expansion is orthogonal to `chunk_context_headers`; both can be enabled together and address different failure modes.

### Design

Add a new index-time processing step that generates synthetic queries per chunk:

1. For each chunk, call an LLM with: `Given this text passage, generate 3-5 questions that this passage would answer. Focus on the vocabulary a user would naturally use when searching for this information.`
2. Append the synthetic queries to the chunk's searchable representation — both for BM25 indexing and embedding generation.
3. Store the synthetic queries separately in metadata for transparency and debugging.

This targets a different failure mode than `chunk_context_headers`: the existing headers add document-level context ("this chunk is from section 3.2 of Source X"), while document expansion adds user-level query patterns (generic example: "how to do procedure Y", "steps for Y", "when to do Y" — NB: the prompt itself must not seed domain vocabulary per Convention 1; these are illustrative only).

### Configuration

```python
@dataclass  
class DocumentExpansionConfig:
    enabled: bool = False
    num_queries: int = 5                             # queries to generate per chunk
    model: LanguageModelConfig | None = None          # LLM for query generation
    include_in_embeddings: bool = True                # append to embedding text
    include_in_bm25: bool = True                      # append to BM25-indexed text
```

Add to `IngestionConfig` as `document_expansion: DocumentExpansionConfig`.

### Implementation Targets

| What | Where | Detail |
|---|---|---|
| Query generation | New BAML function `GenerateSyntheticQueries` | Input: chunk text + optional document context. Output: list of query strings. Prompt MUST be domain-agnostic — no "valves, motors, oil change" examples (Convention 1). Fence both input params; classify in `USER_CONTROLLED_PARAMS` (Convention 3) |
| Chunk augmentation | `IngestionService.ingest()`, after `contextualize_chunks()` | New step: `expand_chunks()`. For each chunk, call LLM, append queries to `ChunkedContent.contextualized` (or a new field) |
| Metadata storage | `ChunkedContent` model | New field: `synthetic_queries: list[str] = field(default_factory=list)`. Stored in vector payload for inspection |
| Embedding integration | `VectorIngestion.ingest()` | When `include_in_embeddings`, append synthetic queries to the text passed to the embedding model |
| BM25 integration | Qdrant BM25 indexing | When `include_in_bm25`, include synthetic queries in the BM25-indexed content field |
| Config wiring | `IngestionConfig` + `IngestionService.__init__()` | Pass config through, call expansion step conditionally. Register `DocumentExpansionConfig` in `_CONFIGS_TO_AUDIT`; bounds-check `num_queries`; add CLAUDE.md bounds entry (Convention 2) |

### Cost Considerations

At ~50 tokens per chunk input and ~100 tokens for 5 generated queries, using Claude Haiku at $0.25/M input + $1.25/M output: ~$0.015 per 100 chunks. A 500-page document with 2000 chunks costs ~$0.30 for document expansion. This is comparable to contextual chunking cost and runs at the same stage.

### Dependencies

- Requires an LLM config (`document_expansion.model`)
- No new external dependencies
- Works with or without vector store (queries can also enhance document store full-text search)

---

## R4. Chunk Position Optimization

**Status:** Not started
**Impact:** Medium — low-effort improvement to generation quality
**Research:** Lost in the Middle (Liu et al., TACL 2024)

### Problem

LLMs exhibit U-shaped attention: they use information at the beginning and end of the context window effectively, but struggle with information in the middle. Liu et al. showed performance drops >20% when relevant information is in the middle vs beginning/end. This holds even for long-context models.

rfnry-rag currently passes chunks to generation in descending score order (highest first) via `chunks_to_context()` in `common/formatting.py`. This means the most relevant chunk is at the beginning (good), but the second-most-relevant is right after it (less good — it would be better at the end), and the least relevant is at the end (wasting the recency-privileged position).

### Design

Add configurable chunk ordering strategies to context assembly:

**A. Score Descending (current default)**
`[1st, 2nd, 3rd, 4th, 5th]` — unchanged behavior.

**B. Primacy-Recency**
`[1st, 3rd, 5th, 4th, 2nd]` — highest-confidence at beginning and end, lowest-confidence in the middle. Directly addresses the Lost-in-the-Middle effect.

**C. Sandwich**
`[1st, 2nd, 5th, 4th, 3rd]` — top two at beginning, remaining in reverse order at end. A compromise between score ordering and position optimization.

### Configuration

```python
class ChunkOrdering(Enum):
    SCORE_DESCENDING = "score_descending"    # current behavior
    PRIMACY_RECENCY = "primacy_recency"      # best at edges, worst in middle
    SANDWICH = "sandwich"                     # top at start, rest reversed at end

# Add to GenerationConfig:
chunk_ordering: ChunkOrdering = ChunkOrdering.SCORE_DESCENDING
```

### Implementation Targets

| What | Where | Detail |
|---|---|---|
| Ordering logic | `common/formatting.py` → `chunks_to_context()` | Add `ordering: ChunkOrdering` parameter. Apply reordering before joining chunks into context string |
| Config wiring | `GenerationConfig` → `GenerationService.__init__()` | Pass ordering to the formatting function. `chunk_ordering` is an enum — no numeric bounds to register — but add a CLAUDE.md note under `GenerationConfig` (Convention 2) |

This is a ~30-line change in `chunks_to_context()` and config wiring. The ordering function itself:

```python
def _reorder_primacy_recency(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Place highest-scored chunks at beginning and end, lowest in middle."""
    if len(chunks) <= 2:
        return chunks
    result = []
    for i, chunk in enumerate(chunks):
        if i % 2 == 0:
            result.insert(len(result) // 2, chunk)  # middle
        else:
            result.append(chunk)  # end
    # Swap: first stays first, last gets second-best
    return [chunks[0]] + result[1:-1] + [chunks[1]] if len(chunks) > 2 else chunks
```

### Dependencies

None. Pure logic change.

---

## R5. Adaptive Retrieval Parameters

**Status:** Not started
**Impact:** Medium — right-sizes retrieval per query
**Research:** LaRA (ICML 2025), Google DeepMind SELF-ROUTE

### Problem

rfnry-rag uses static `top_k` and fixed method weights for every query. LaRA showed that optimal parameters vary by query type:

- Simple factual queries (who/where/when): fewer chunks, vector + BM25 dominant
- Comparative queries: more chunks, broader retrieval methods
- Multi-hop queries: more chunks, graph retrieval dominant
- Hallucination-prone queries: fewer chunks, higher precision focus

Additionally, the 72B model in LaRA performed better with more chunks, while the 7B model peaked at intermediate counts then degraded. The optimal top_k depends on both the query and the model.

### Design

Three adaptive mechanisms, each independently valuable:

**A. Dynamic Top-K**

Estimate query complexity before retrieval and adjust top_k accordingly:

1. **Heuristic classification** (no LLM call): query length, question word type (who/what/how/compare), presence of specific identifiers, number of entities mentioned. Fast, free, catches the obvious cases.
2. **LLM classification** (optional, uses `RetrievalConfig.enrich_lm_client`): lightweight call to classify query as `simple | moderate | complex`. More accurate, adds one LLM round-trip.

Map complexity to top_k within configured bounds:
- `simple` → `top_k_min` (default 3)
- `moderate` → `top_k` (configured default, currently 5)
- `complex` → `top_k_max` (default 15)

**B. Task-Aware Method Weighting**

Use query classification to adjust per-method weights at query time:

| Query Type | Vector Weight | Document Weight | Graph Weight | Tree Weight |
|---|---|---|---|---|
| Factual (who/where/when) | 1.2x | 0.8x | 0.8x | 0.8x |
| Comparative | 0.8x | 1.2x | 0.8x | 1.2x |
| Entity-relationship | 0.8x | 0.8x | 1.5x | 0.8x |
| Procedural (how-to) | 1.0x | 1.2x | 0.8x | 1.2x |

These are multipliers on the base `method.weight`. Configurable via a weight profile map.

**C. Confidence-Based Expansion**

If retrieval returns low-confidence results (max score below `grounding_threshold`), progressively expand:

1. Increase top_k by 2x and retry
2. If still low, apply a different query rewriter (if available) and retry
3. If still low and corpus fits in context, escalate to Direct Context mode (requires R1)

This creates a self-healing retrieval loop that tries harder before giving up.

### Configuration

```python
@dataclass
class AdaptiveRetrievalConfig:
    enabled: bool = False
    top_k_min: int = 3
    top_k_max: int = 15
    use_llm_classification: bool = False   # False = heuristic only
    task_weight_profiles: dict[str, dict[str, float]] | None = None  # override default profiles
    confidence_expansion: bool = False     # enable progressive expansion
    max_expansion_retries: int = 2
```

Add to `RetrievalConfig` as `adaptive: AdaptiveRetrievalConfig`.

### Implementation Targets

| What | Where | Detail |
|---|---|---|
| Query classifier (heuristic) | New module: `modules/retrieval/search/classification.py` | Rule-based: regex for question words, entity count, query length → complexity enum |
| Query classifier (LLM) | New BAML function `ClassifyQueryComplexity` | Input: query text (fenced + classified per Convention 3). Output: `{complexity: simple\|moderate\|complex, query_type: factual\|comparative\|entity\|procedural}`. Both fields are *intent*-based — universal across domains. Do NOT seed domain examples ("electrical query", "legal contract") in the prompt (Convention 1) |
| Dynamic top_k | `RetrievalService.retrieve()` | Before dispatch, classify query, compute effective top_k from bounds |
| Weight adjustment | `RetrievalService.retrieve()` | Before RRF fusion, multiply method weights by task profile |
| Confidence expansion | `RagEngine.query()` around `_retrieve_chunks()` | Wrap retrieval in retry loop. Check max score against threshold. Expand parameters on retry |
| Config wiring | `RetrievalConfig` + `RagEngine.initialize()` | Pass adaptive config to RetrievalService. Register `AdaptiveRetrievalConfig` in `_CONFIGS_TO_AUDIT`; bounds-check `top_k_min`, `top_k_max`, `max_expansion_retries`; add CLAUDE.md bounds entry (Convention 2) |

### Dependencies

- Heuristic classification: no dependencies
- LLM classification: requires `RetrievalConfig.enrich_lm_client` (already exists)
- Confidence expansion: benefits from R1 (Direct Context fallback) but works independently (can just increase top_k without LC fallback)

---

## R6. Multi-Hop Iterative Retrieval

**Status:** Not started
**Impact:** Medium — addresses RAG's biggest failure mode
**Research:** Google DeepMind SELF-ROUTE failure analysis

### Problem

SELF-ROUTE's error analysis classified RAG failures into four categories. The dominant one: **multi-step reasoning** — queries like "What nationality is the performer of song X?" that require chaining retrieval (find song → find performer → find nationality). rfnry-rag does single-pass retrieval: one query, one search, one set of results. Multi-hop queries fail because no single chunk contains the full answer chain.

Query rewriting (HyDE, multi-query, step-back) partially helps by generating query variants, but all variants still execute in parallel against the same corpus. They don't chain — results from one search don't inform the next.

### Design

A new query processing mode that decomposes complex queries into sequential retrieval steps:

1. **Decomposition:** LLM breaks the original query into sub-questions. "What nationality is the performer of Bohemian Rhapsody?" → ["Who performed Bohemian Rhapsody?", "What nationality is [result from step 1]?"]
2. **Sequential execution:** Each sub-question triggers a full retrieval pass (search + fusion + optional reranking). Results from each step are accumulated and available as context for the next step's query rewriting.
3. **Synthesis:** After all steps, the accumulated context is passed to generation along with the original query.

This is structurally similar to chain-of-thought prompting but applied to the retrieval phase.

### Configuration

```python
@dataclass
class IterativeRetrievalConfig:
    enabled: bool = False
    max_hops: int = 3                                  # max retrieval iterations
    decomposition_model: LanguageModelConfig | None = None  # LLM for query decomposition
    min_complexity: str = "moderate"                    # only decompose queries at this complexity or above
```

Add to `RetrievalConfig` as `iterative: IterativeRetrievalConfig`.

### Implementation Targets

| What | Where | Detail |
|---|---|---|
| Query decomposer | New BAML function `DecomposeQuery` | Input: query + optional accumulated context (both user-controlled — fence + classify per Convention 3). Output: `{sub_questions: list[str], reasoning: str}` or `{answerable: true}` if no decomposition needed. Prompt stays domain-agnostic (Convention 1) |
| Iterative loop | New service: `modules/retrieval/search/iterative.py` | `IterativeRetrievalService` wrapping `RetrievalService`. Calls decompose → retrieve → accumulate → repeat |
| Context accumulation | Within iterative service | Maintain `accumulated_chunks: list[RetrievedChunk]` across hops. Deduplicate by chunk_id |
| Integration | `RagEngine._retrieve_chunks()` | When iterative retrieval is enabled and query complexity >= threshold, use `IterativeRetrievalService` instead of direct `RetrievalService.retrieve()` |
| Complexity check | Reuse R5's query classifier | If R5 is implemented, reuse classification. Otherwise, always decompose (LLM decides if decomposition is needed via the `answerable` escape hatch) |
| Config wiring | `RetrievalConfig` | Register `IterativeRetrievalConfig` in `_CONFIGS_TO_AUDIT`; bounds-check `max_hops`; add CLAUDE.md bounds entry (Convention 2) |

### Interaction with Query Rewriting

Iterative retrieval and query rewriting are orthogonal:

- **Query rewriting** (HyDE, multi-query, step-back) generates alternative formulations of the *same* question. Parallel execution.
- **Iterative retrieval** decomposes a complex question into *sequential* sub-questions. Serial execution with context accumulation.

Both can be active simultaneously: each hop in the iterative loop can use query rewriting for its sub-question.

### Dependencies

- Requires an LLM config for decomposition
- Benefits from R5 (query complexity classification for gating) but works independently
- No new external dependencies

---

## R7. Local Cross-Encoder Reranking

**Status:** Not started
**Impact:** Medium — best generalization, removes API dependency
**Research:** BEIR (cross-encoder reranking wins on 16/18 datasets)

### Problem

BEIR demonstrated that BM25 + cross-encoder reranking is the best-generalizing retrieval approach, winning on 16 of 18 heterogeneous datasets. rfnry-rag supports reranking via Cohere and Voyage APIs. These work well but add:

- **Latency:** External API round-trip (100-500ms per rerank call)
- **Cost:** Per-query pricing for reranking API calls
- **Dependency:** External service availability
- **Privacy:** Sending content to third-party reranking APIs

Local cross-encoder models (sentence-transformers, BGE-reranker, MiniLM) provide the same cross-attention generalization benefit without these trade-offs.

### Design

A new reranking implementation using local cross-encoder models:

```python
class LocalCrossEncoderReranking:
    """Implements BaseReranking using a local cross-encoder model."""
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",  # default model
        device: str = "cpu",           # "cpu", "cuda", "mps"
        batch_size: int = 32,
    ): ...
    
    async def rerank(
        self,
        query: str,
        results: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]: ...
```

The implementation loads a `CrossEncoder` from sentence-transformers and scores each (query, chunk.content) pair. Since cross-encoder inference is CPU/GPU-bound (not I/O-bound), wrap it in `asyncio.to_thread()` to avoid blocking the event loop.

### Configuration

```python
from rfnry_rag.retrieval import LocalCrossEncoderReranking

config = RagServerConfig(
    retrieval=RetrievalConfig(
        reranker=LocalCrossEncoderReranking(
            model_name="BAAI/bge-reranker-v2-m3",
            device="cuda",
        ),
    ),
    # ...
)
```

No new config dataclass needed — this is a new implementation of the existing `BaseReranking` protocol. Users swap it in where they currently use `CohereReranking` or `VoyageReranking`.

### Implementation Targets

| What | Where | Detail |
|---|---|---|
| Cross-encoder wrapper | New file: `modules/retrieval/search/reranking/local.py` | `LocalCrossEncoderReranking` implementing `BaseReranking` protocol |
| Model loading | `__init__()` | Load model on initialization. Support lazy loading (load on first rerank call) to avoid startup cost if reranking isn't always used |
| Async bridge | `rerank()` | `await asyncio.to_thread(self._rerank_sync, query, results, top_k)` |
| Batch scoring | `_rerank_sync()` | Score all (query, chunk) pairs in batches. Sort by score. Return top_k |
| Optional dependency | `pyproject.toml` | Add `sentence-transformers` as optional dependency under `[local-reranking]` extra |

### Performance Characteristics

Based on BEIR benchmarks with 1M documents:

- **API reranking (Cohere):** ~450ms per query (network + compute), reranks top-100
- **Local cross-encoder (MiniLM):** ~350ms per query on GPU, ~6100ms on CPU, reranks top-100
- **Local cross-encoder (BGE-reranker-v2):** Larger model, better accuracy, ~2x MiniLM latency

For typical rfnry-rag usage (reranking top-20 to top-50 chunks, not 100), local GPU latency will be well under 200ms. CPU-only deployments should limit rerank candidates to top-30.

### Dependencies

- `sentence-transformers` (optional dependency)
- `torch` (transitive via sentence-transformers)
- No new external API dependencies — that's the point

---

## R8. Retrieval Evaluation & Diagnostics

**Status:** Not started
**Impact:** Medium — debuggability, user confidence, competitive differentiation
**Research:** Akita (2026), BEIR

### Problem

Akita's strongest argument against RAG tooling is opacity: "when BM25 misses, you know why (the word is not there). When a vector DB returns garbage, you get a plausible-but-wrong chunk with zero diagnostic signal."

rfnry-rag has evaluation metrics (EM, F1, LLMJudge, retrieval recall/precision) but these are batch metrics — they tell you aggregate quality, not why a specific query failed. Users debugging retrieval quality need per-query traces and failure analysis.

### Design

Phase C12 shipped `GraphRetrieval.trace(entity_name, max_hops, relation_types, knowledge_id) -> list[GraphPath]` as a per-method diagnostic helper — the pattern to generalize here. That method already demonstrates timing capture, optional-filter semantics, and empty-list-on-error behavior; extend the same discipline across every retrieval method so the pipeline-level trace in R8 has consistent data to aggregate.

Three capabilities:

**A. Retrieval Trace**

For each query, capture the full retrieval pipeline state:

```python
@dataclass
class RetrievalTrace:
    query: str
    rewritten_queries: list[str]              # after query rewriting
    per_method_results: dict[str, list[ScoredChunk]]  # raw results per method, pre-fusion
    fused_results: list[ScoredChunk]          # after RRF
    reranked_results: list[ScoredChunk] | None  # after reranking
    refined_results: list[ScoredChunk] | None   # after chunk refinement
    final_results: list[RetrievedChunk]       # what went to generation
    grounding_decision: str                    # "grounded" | "ungrounded" | "clarification"
    confidence: float
    timings: dict[str, float]                 # per-stage latency
    routing_decision: str | None              # "direct" | "retrieval" | "hybrid_rag" | "hybrid_lc" (if R1)
```

Return this alongside `QueryResult` when requested (opt-in flag to avoid overhead in production).

**B. Failure Classification**

When grounding fails (ungrounded response), classify why:

| Failure Type | Detection | Meaning |
|---|---|---|
| `vocabulary_mismatch` | BM25 returned 0 results, vector returned low-score results | Query terms don't match document terms |
| `chunk_boundary` | High-score chunk exists but answer spans chunk boundary | Chunking split the answer |
| `scope_miss` | No results in target `knowledge_id`, results exist in other scopes | Wrong knowledge scope |
| `entity_not_indexed` | Graph retrieval returned nothing, query contains named entities | Entity not extracted during ingestion |
| `low_relevance` | All methods returned results but all scores below threshold | Content exists but isn't relevant to query |
| `insufficient_context` | Relevance gate passed but generation grounding failed | Retrieved chunks lack enough context for answer |

Classification can be heuristic (check which methods returned what) for most cases, with optional LLM classification for ambiguous failures.

**C. Benchmark Harness**

A structured way to evaluate retrieval quality against a test set:

```python
@dataclass
class BenchmarkCase:
    query: str
    expected_answer: str
    expected_source_ids: list[str] | None = None  # optional: which sources should be retrieved

@dataclass  
class BenchmarkReport:
    total_cases: int
    retrieval_recall: float        # fraction of cases where expected source was in top-k
    retrieval_precision: float     # fraction of top-k chunks that were relevant
    generation_em: float           # exact match on answers
    generation_f1: float           # token F1 on answers
    llm_judge_score: float | None  # optional LLM judge score
    per_case_traces: list[RetrievalTrace]  # full traces for debugging
    failure_distribution: dict[str, int]   # counts per failure type
```

Expose via `RagEngine.benchmark(cases: list[BenchmarkCase]) -> BenchmarkReport` and via CLI: `rfnry-rag retrieval benchmark test_cases.json -k knowledge_id`.

### Implementation Targets

| What | Where | Detail |
|---|---|---|
| Trace collection | `RetrievalService.retrieve()` | Add optional `trace: bool` parameter. When True, collect intermediate results at each stage into `RetrievalTrace` |
| Trace pass-through | `RagEngine.query()` → `_retrieve_chunks()` | Accept `trace=True`, return `RetrievalTrace` alongside results |
| QueryResult extension | `common/models.py` → `QueryResult` | New optional field: `trace: RetrievalTrace \| None = None` |
| Failure classifier | New module: `modules/evaluation/failure_analysis.py` | Input: query + RetrievalTrace. Output: failure type + reasoning |
| Benchmark runner | `modules/evaluation/benchmark.py` | Iterate cases, call `query()` with trace, compute metrics, aggregate |
| CLI integration | `cli/commands/` | New command: `rfnry-rag retrieval benchmark` |
| Output formatting | `cli/formatters/` | Trace and benchmark report formatters |

### Dependencies

- Trace collection: no new dependencies (adds overhead only when enabled)
- Failure classification: heuristic mode has no dependencies; LLM mode needs an LM config
- Benchmark: uses existing evaluation metrics (EM, F1, LLMJudge)

---

## Implementation Order

The features have natural dependencies and synergies:

```
Phase 1 (Foundation)
├── R4. Chunk Position Optimization      ← smallest scope, immediate value
├── R3. Document Expansion at Index Time ← index-time improvement, no runtime changes
└── R7. Local Cross-Encoder Reranking    ← drop-in replacement, no architecture changes

Phase 2 (Intelligence)
├── R8. Retrieval Diagnostics            ← trace infrastructure; observability prerequisite for R1's AUTO routing + R5's adaptive weights
├── R1. Context-Aware Routing            ← token counting infra partially done (Phase A's tiktoken + count_tokens)
└── R5. Adaptive Retrieval Parameters    ← requires query classification; shares classifier with R6

Phase 3 (Advanced)
├── R2. RAPTOR Summarization Retrieval   ← new ingestion + retrieval method pair
└── R6. Multi-Hop Iterative Retrieval    ← benefits from R5's query classifier, R1's LC fallback
```

Phase 1 features are independent, small-scoped, and immediately valuable. They can ship together or separately.

Phase 2 features build the intelligence layer — observability, routing, and adaptation. R8 moves to the front because R1's `AUTO` mode and R5's adaptive parameters are unobservable without trace infrastructure; tuning either blind wastes weeks. R1 remains the most important feature overall but genuinely benefits from having R8 land first. R5's classifier is shared with R6, so landing R5 first reduces R6's implementation cost.

Phase 3 features are the most complex and benefit from the infrastructure built in Phases 1-2. RAPTOR needs the embedding and vector store infrastructure to be well-tested. Iterative retrieval benefits from query classification (R5) and can use LC fallback (R1) as an escape hatch.

---

## What This Roadmap Does NOT Include

Features explicitly excluded based on research findings:

1. **More embedding providers** — Diminishing returns. BEIR and LaRA show the gap between embedding models is small compared to the gap between retrieval strategies.

2. **Agent/chain orchestration** — Not rfnry-rag's domain. We're a retrieval engine, not a workflow framework.

3. **Fine-tuned dense retrievers** — BEIR showed these collapse out-of-domain. Hybrid retrieval with general models is more robust.

4. **Advanced chunking strategies** — Context windows are growing faster than corpora. The trend favors larger context over smarter chunking. R1 (Direct Context) is the answer, not chunking tricks.

5. **Streaming reranking / streaming fusion** — Premature optimization. Get the retrieval quality right first.
