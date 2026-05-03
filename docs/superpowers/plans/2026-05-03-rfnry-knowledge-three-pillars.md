# Three-pillars rename implementation plan

> **For agentic workers:** Use superpowers:executing-plans (inline). Steps use checkbox (`- [ ]`).

**Goal:** Rename retrieval/ingestion methods to Semantic/Keyword/Entity pillars; split BM25 out of VectorRetrieval; fold DocumentRetrieval into KeywordRetrieval; drop StructuredRetrieval; rename `QueryMode.INDEXED → RETRIEVAL`, `FULL_CONTEXT → DIRECT`.

**Working dir:** `/home/frndvrgs/software/rfnry/knowledge/packages/python`

---

## T1 — Inspect current call sites

- [ ] Map every reference to be renamed: run a grep matrix (VectorRetrieval/DocumentRetrieval/GraphRetrieval/StructuredRetrieval, VectorIngestion/DocumentIngestion/GraphIngestion, QueryMode.INDEXED/FULL_CONTEXT, retrieval/enrich/, retrieval/methods/enrich.py).

## T2 — `QueryMode` rename

- [ ] `config/routing.py`: `INDEXED → RETRIEVAL`, `FULL_CONTEXT → DIRECT`. Update docstrings.
- [ ] `knowledge/engine.py`: every `QueryMode.INDEXED` → `QueryMode.RETRIEVAL`; `QueryMode.FULL_CONTEXT` → `QueryMode.DIRECT`. Update telemetry row `mode=` strings (`"indexed"` → `"retrieval"`; `"full_context"` → `"direct"`).
- [ ] All tests: same swap.

## T3 — Split BM25 out of `VectorRetrieval`; rename → `SemanticRetrieval`

- [ ] Read `retrieval/methods/vector.py` carefully. Identify dense-only paths and BM25-only paths.
- [ ] Create `retrieval/methods/semantic.py` containing `SemanticRetrieval` — dense + (optional sparse hybrid) only. Drop all `bm25_*` fields and `_keyword_search` / `_bm25_*` methods.
- [ ] Save the BM25 logic for T4.
- [ ] Delete `retrieval/methods/vector.py`.

## T4 — Create `KeywordRetrieval`; fold in `DocumentRetrieval`

- [ ] Read `retrieval/methods/document.py`.
- [ ] Create `retrieval/methods/keyword.py` with class `KeywordRetrieval`:
    - Fields: `backend: Literal["bm25", "postgres_fts"] = "bm25"`, `weight=1.0`, `top_k=None`, `vector_store`, `document_store`, `bm25_max_chunks`, `bm25_max_indexes`, `parent_expansion`, `use_substring_fallback`.
    - `__post_init__` validates the right store is supplied per backend.
    - `search()` dispatches on `self.backend`. Body lifts BM25 code from old `VectorRetrieval` and FTS+substring code from old `DocumentRetrieval`.
- [ ] Delete `retrieval/methods/document.py`.

## T5 — Rename `GraphRetrieval` → `EntityRetrieval`

- [ ] Create `retrieval/methods/entity.py` with `class EntityRetrieval(BaseRetrievalMethod):` (body lifted from `retrieval/methods/graph.py`, name updated). Inline any cross-reference logic from `retrieval/enrich/` that still applies (the consumer asked for entity-pillar to absorb it where useful).
- [ ] Delete `retrieval/methods/graph.py`.

## T6 — Drop StructuredRetrieval and `retrieval/enrich/`

- [ ] Delete `retrieval/methods/enrich.py`.
- [ ] Delete `retrieval/enrich/` directory entirely.
- [ ] Engine `query()` plumbing: drop the `_structured_retrieval` field and the special-case dispatch; the parallel-pillar fusion path is the only retrieval path now.

## T7 — Rename ingestion methods

- [ ] `ingestion/methods/vector.py` → `ingestion/methods/semantic.py`; class `VectorIngestion` → `SemanticIngestion`.
- [ ] `ingestion/methods/document.py` → `ingestion/methods/keyword.py`; class `DocumentIngestion` → `KeywordIngestion`.
- [ ] `ingestion/methods/graph.py` → `ingestion/methods/entity.py`; class `GraphIngestion` → `EntityIngestion`.
- [ ] Delete the old files.
- [ ] Update `ingestion/methods/__init__.py` re-exports.
- [ ] Update `AnalyzedIngestion` and `DrawingIngestion` for any internal references (e.g. `isinstance(m, DocumentIngestion)` → `isinstance(m, KeywordIngestion)`).

## T8 — Update top-level `__init__.py`

- [ ] Drop old re-exports: `VectorRetrieval`, `DocumentRetrieval`, `GraphRetrieval`, `StructuredRetrieval`, `VectorIngestion`, `DocumentIngestion`, `GraphIngestion`.
- [ ] Add new re-exports: `SemanticRetrieval`, `KeywordRetrieval`, `EntityRetrieval`, `SemanticIngestion`, `KeywordIngestion`, `EntityIngestion`.
- [ ] Update `__all__`.

## T9 — Update `RetrievalConfig`

- [ ] If `cross_reference_enrichment` is still consumed (was used by StructuredRetrieval), thread it into `EntityRetrieval` config; otherwise leave the field but mark as consumed by `EntityRetrieval`. Same name on the config keeps backward-shape but the meaning is now "entity pillar enriches with cross-refs".

## T10 — Update tests

- [ ] Run `grep -rln 'VectorRetrieval\|DocumentRetrieval\|GraphRetrieval\|StructuredRetrieval\|VectorIngestion\|DocumentIngestion\|GraphIngestion\|QueryMode\.INDEXED\|QueryMode\.FULL_CONTEXT\|from rfnry_knowledge.retrieval.enrich\|from rfnry_knowledge.retrieval.methods.enrich' tests/`.
- [ ] For each: rename to the new symbol. For tests asserting BM25 behavior on `VectorRetrieval`, switch the test to construct a `KeywordRetrieval(backend="bm25", vector_store=...)`. For tests asserting FTS on `DocumentRetrieval`, switch to `KeywordRetrieval(backend="postgres_fts", document_store=...)`.
- [ ] For tests that exercised `StructuredRetrieval` specifically, either re-target to `EntityRetrieval` (if the cross-ref enrichment is still there) or delete them.
- [ ] Adjust telemetry-row `mode=` string assertions.

## T11 — mypy + ruff + tests

- [ ] `uv run poe typecheck`
- [ ] `uv run poe check`
- [ ] `uv run poe format`
- [ ] `uv run poe test`

## T12 — Docs

- [ ] `packages/python/CLAUDE.md`: update the architecture/retrieval-pipeline sections, the `Config defaults and enforced bounds` table, and the entry-points example to use `SemanticRetrieval` / `KeywordRetrieval` / `EntityRetrieval` and `QueryMode.RETRIEVAL` / `QueryMode.DIRECT`.
- [ ] `packages/python/README.md`: same swap; update the example snippet so it shows the three pillars composed.

## T13 — Final grep audit

- [ ] All of these grep clean (no matches in `src/` or `tests/`):
    - `VectorRetrieval`, `DocumentRetrieval`, `GraphRetrieval`, `StructuredRetrieval`
    - `VectorIngestion`, `DocumentIngestion`, `GraphIngestion`
    - `QueryMode\.INDEXED`, `QueryMode\.FULL_CONTEXT`
    - `from rfnry_knowledge.retrieval.enrich`, `from rfnry_knowledge.retrieval.methods.enrich`
    - `retrieval/methods/(vector|document|graph)\.py` paths

## T14 — Commit + push to main

- [ ] Single conventional `refactor!:` commit summarizing all renames.
- [ ] `git push origin main`.
