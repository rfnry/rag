# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Resolves 45 findings from the 2026-04-23 comprehensive review across
correctness, security, operational safety, and hardening. 25 commits; 689
tests passing (up from 629).

### 2026-04-27 R8.3 — Benchmark harness + CLI

R8.1 captured traces; R8.2 classified failures; R8.3 ships the
user-facing payoff: aggregate metrics PLUS per-case traces PLUS
failure-class distribution in one report. New `RagEngine.benchmark(cases)
-> BenchmarkReport` (Python API) and `rfnry-rag benchmark cases.json -k
<knowledge_id> [-o report.json]` (CLI). Closes the R8 phase by exposing
trace + failure-distribution aggregation as a single user-facing report,
which gates Phase 2 R1/R5 on observability before tuning.

Public API (re-exported from `rfnry_rag.retrieval`):

- `BenchmarkCase(query, expected_answer, expected_source_ids=None)` —
  one evaluation case.
- `BenchmarkConfig(concurrency=1, failure_threshold=0.5)` — runner
  knobs. `concurrency` bounded `[1, 20]`; `failure_threshold` bounded
  `[0.0, 1.0]`. Both validated in `__post_init__`; registered in the
  bounds-contract test (Convention 2).
- `BenchmarkCaseResult(case, result, failure, metrics)` — per-case
  detail. `result` is the full `QueryResult` (with `trace` from R8.1);
  `failure` is `FailureClassification | None` from R8.2; `metrics`
  carries per-case `em`, `f1`, optional `retrieval_recall`,
  `retrieval_precision`, `llm_judge`.
- `BenchmarkReport(total_cases, retrieval_recall, retrieval_precision,
  generation_em, generation_f1, llm_judge_score, failure_distribution,
  per_case_results)` — aggregate. `retrieval_recall` /
  `retrieval_precision` are `None` when at least one case omits
  `expected_source_ids` (N/A is distinct from 0.0).
  `failure_distribution` is a `dict[str, int]` keyed on
  `FailureType.name` (`"VOCABULARY_MISMATCH"`, etc).
- `run_benchmark(cases, query_fn, config=None, llm_judge=None)` — the
  underlying orchestrator; takes a callable so tests can stub without
  a full engine.

`failure_distribution` keys on `FailureType.name` rather than
`FailureClassification`: the dataclass is `frozen=True` but contains a
`dict[str, ...]` field, so Python does not generate `__hash__` for it
and using it as a `Counter` key would raise `TypeError: unhashable type`
at runtime. Keying on the name also makes `--output report.json`
human-readable without post-processing.

Failure rule: F1 strictly below `failure_threshold` (default 0.5 — an
honest mid-point: weaker is too generous; stricter mis-classifies
legitimate paraphrases) OR `trace.grounding_decision == "ungrounded"`.
The second clause catches the small-but-real "F1 was technically high
but grounding still flagged it ungrounded" category.

CLI: `rfnry-rag benchmark CASES_FILE -k <knowledge_id> [-o report.json]
[-c concurrency] [--failure-threshold T]`. Exit code 0 on success
regardless of failure rate (treating failure-rate as a CI signal is a
follow-up). Pretty stdout summary shows totals, EM, F1, optional LLM
judge, and the per-failure-type histogram.

Reuses the existing `ExactMatch` and `F1Score` from `metrics.py`, and
(when configured) `LLMJudgment` for per-case judging. Adds a
source-id-based `retrieval_recall` / `retrieval_precision` computation
distinct from the content-based `RetrievalRecall` / `RetrievalPrecision`
in `retrieval_metrics.py` — this measures whether the benchmark's
`expected_source_ids` were retrieved, not chunk-content overlap with the
expected answer. No new metric implementations and no new LLM calls
beyond what the configured metrics already do.

Cost / latency shape (consumer-visible):

- Cost: each case runs one full `engine.query`, so a benchmark of N
  cases consumes N generation LLM calls. Passing
  `llm_judge=LLMJudgment(...)` doubles LLM calls (1 generation + 1 judge
  per case). Failed-case classification (R8.2) is heuristic-only — no
  extra LLM calls.
- Latency: 1 000 cases serially (default `concurrency=1`) at ~2 s per
  `query` runs ~30 minutes. Bound `BenchmarkConfig.concurrency` (max 20)
  for parallelism — the same `run_concurrent` helper R3 uses.

8 tests added (6 unit + 1 CLI smoke + 1 LLM-judge unit); test count
1052 -> 1060.

### 2026-04-27 R8.2 — Heuristic failure classification

R8.1 produced traces; R8.2 turns them into actionable diagnostics. New
pure-inspection function `classify_failure(query, trace) ->
FailureClassification` reads a `RetrievalTrace` and returns one of seven
`FailureType` verdicts (`VOCABULARY_MISMATCH`, `CHUNK_BOUNDARY`,
`SCOPE_MISS`, `ENTITY_NOT_INDEXED`, `LOW_RELEVANCE`,
`INSUFFICIENT_CONTEXT`, `UNKNOWN`) so a benchmark report can answer
"which class of failure dominates my workload" without paying per-case
LLM cost. Without this, "your retrieval failed" stays opaque; with it,
"40% of failures are vocabulary_mismatch — enable R3 expansion to fix
the dominant class" becomes a one-line diagnostic. R8.3's benchmark
harness consumes it.

Public API:

- New module `retrieval/modules/evaluation/failure_analysis.py` exporting
  `FailureType`, `FailureClassification`, and `classify_failure`. All
  three re-exported from `rfnry_rag.retrieval.modules.evaluation`.
- `FailureClassification` carries `type: FailureType`, `reasoning: str`
  (one-sentence explanation), and `signals: dict[str, str | float | int
  | bool | None]` (the trace-derived values that drove the verdict,
  including the threshold compared against — for transparency and for
  downstream consumers that want to override). Tightened from the
  initial `dict[str, Any]` in polish commit `c37dc8d` to keep the
  payload JSON-serializable for R8.3's report output.
- `FailureClassification` is `frozen=True`. The dataclass contains a
  `dict` field, so it is NOT auto-hashable — that's why R8.3's
  failure-distribution Counter keys on `FailureType.name` (string),
  not on the dataclass itself.
- Heuristic-only first pass: no LLM calls, no new dependencies. An
  LLM-backed classifier is a documented follow-up.

Heuristic priority (first-match wins; documented in module + function
docstrings, with test #8 as regression guard): `VOCABULARY_MISMATCH`
beats `CHUNK_BOUNDARY` beats `SCOPE_MISS` beats `ENTITY_NOT_INDEXED`
beats `LOW_RELEVANCE` beats `INSUFFICIENT_CONTEXT` beats `UNKNOWN`.
Three module-private threshold constants — `_VOCABULARY_MISMATCH_THRESHOLD`
(0.4), `_HIGH_RELEVANCE_THRESHOLD` (0.7), `_LOW_RELEVANCE_THRESHOLD`
(0.3) — live at module scope, not on a config dataclass. They are
heuristic defaults; if a consumer needs to tune them, promote to a
`FailureAnalysisConfig` then.

`ENTITY_NOT_INDEXED` uses a generic capitalized-multi-char-token regex
(`\b[A-Z][A-Z0-9_-]{2,}\b`) — matches `R-101`, `PumpModelX`,
`EntityXYZ`, but also `JSON`, `HTTP`. The matched token is reported in
`signals["matched_token"]` so consumers can judge false positives.
Convention 1 (consumer-agnostic): no domain vocabulary baked into any
literal.

Caller's responsibility to invoke only on failed cases — R8.3's
benchmark calls it on every case where the produced answer doesn't
match the expected. Test count 1044 → 1052 (+8 unit tests in
`test_failure_classification.py`).

### 2026-04-27 R8.1 — RetrievalTrace dataclass + opt-in trace=True flag

Vector retrieval is opaque: when an answer is wrong, there's no evidence
of which stage drifted. R8.1 adds a `RetrievalTrace` dataclass capturing
the full per-query pipeline state (rewritten queries, per-method raw
results, fused, reranked, refined, final, grounding decision, confidence,
per-stage timings) so R8.2 (failure classification) and R8.3 (benchmark
harness) have something concrete to read. Tuning R1's AUTO routing or
R5's adaptive weights without this would be blind.

Public API:

- New `RetrievalTrace` dataclass in `retrieval/common/models.py` next to
  `RetrievedChunk`. 12 fields total: `query`, `rewritten_queries`,
  `per_method_results`, `fused_results`, `reranked_results`,
  `refined_results`, `final_results`, `grounding_decision`, `confidence`,
  `routing_decision` (R1 placeholder), `timings`, plus `knowledge_id`
  (R8.2 coordination — used to detect SCOPE_MISS). Constructible with
  just `query=...`; every other field has a safe default.
- `QueryResult.trace: RetrievalTrace | None = None` — new optional field.
  Existing consumers unaffected.
- `RetrievalService.retrieve(...)` signature change:
  `-> list[RetrievedChunk]` becomes
  `-> tuple[list[RetrievedChunk], RetrievalTrace | None]`. New
  `trace: bool = False` parameter. Default path is byte-for-byte
  unchanged: returns `(chunks, None)` and skips all collection logic.
  Internal callers in `RagEngine._retrieve_chunks` and the test suite
  unpack the tuple.
- `RagEngine.query(..., trace: bool = False)` threads the flag through
  retrieval, populates `grounding_decision` / `confidence` post-grounding,
  and attaches the trace to `QueryResult.trace`. `query_stream` is out
  of scope — streaming-mode trace is deferred.

`None` vs `[]` distinction is load-bearing: `reranked_results is None`
iff the reranker is not configured (same shape for `refined_results`).
`per_method_results` always includes every declared method as a key, with
`[]` for "ran and produced nothing". Conflating these would erase the
signal R8.2's classifiers depend on.

No new dependencies, no new BAML, no DB persistence. Test count
1035 → 1044 (+9 retrieval-trace unit tests in `test_retrieval_trace.py`).

### 2026-04-26 R3 — Document expansion at index time (synthetic queries per chunk)

Bridge the user-vocabulary-vs-document-vocabulary gap with docT5query-style
expansion: opt-in LLM-generated synthetic questions per chunk, folded into
embedding text and BM25 text at index time. BEIR (Thakur et al., NeurIPS
2021) shows this beats vanilla BM25 on 11 of 18 datasets while preserving
BM25's generalization.

Public API:

- New `DocumentExpansionConfig` dataclass (next to `IngestionConfig` in
  `server.py`). Defaults: `enabled=False`, `num_queries=5`,
  `lm_client=None`, `include_in_embeddings=True`, `include_in_bm25=True`,
  `concurrency=5`. Bounds: `num_queries` ∈ [1, 20], `concurrency` ∈
  [1, 100]. `enabled=True` requires `lm_client` (no opinionated default
  model — consumer chooses).
- `IngestionConfig.document_expansion: DocumentExpansionConfig` (default
  factory; consumers opt in by setting `enabled=True` + providing
  `lm_client`).
- `ChunkedContent.synthetic_queries: list[str]` field (always populated
  when expansion ran; empty otherwise — useful for debugging / audit).
- `ChunkedContent.text_for_embedding(*, include_synthetic: bool = True)`
  and `text_for_bm25(*, include_synthetic: bool = True)` — explicit
  gating methods. The `embedding_text` property remains backward-
  compatible (delegates to `text_for_embedding(include_synthetic=True)`).
- New BAML function `GenerateSyntheticQueries(passage, num_queries) ->
  SyntheticQueries` (wrapper class with `queries: string[]`, mirroring
  the existing `QueryVariants` shape). Both string params fenced;
  classified in `USER_CONTROLLED_PARAMS`. Prompt body audited
  domain-agnostic.
- New helper `expand_chunks(chunks, config, registry)` in
  `modules/ingestion/chunk/expand.py` (sibling to `chunk/context.py`,
  Convention 4). Bounded concurrency via `run_concurrent`. LLM failures
  raise `IngestionError` with the offending `chunk_index`.

Cost / latency shape (consumer-visible):

- Cost: enabling expansion adds N LLM calls per ingest where N = chunk
  count. With `num_queries=5` and a typical 800-token model context,
  expect roughly 800 input + 200 output tokens per call.
- Latency: at the default `concurrency=5`, a 2 000-chunk file takes
  ~400 round-trips × ~500 ms ≈ 3 minutes for the expansion step alone.
  Tune `concurrency` upward if your provider tier permits it.

Implementation notes:

- `expand_chunks` always stores `synthetic_queries` on the chunk;
  `include_in_embeddings` and `include_in_bm25` gate **what gets sent to
  the embedding model / BM25 indexer**, not whether queries are stored.
  This keeps the `ChunkedContent` shape predictable and lets consumers
  inspect queries even when both flags are `False`.
- `VectorIngestion` now accepts `include_synthetic_in_embeddings` /
  `include_synthetic_in_bm25` constructor kwargs; `RagEngine` threads
  them through from `IngestionConfig.document_expansion`.
- Vector payload carries `synthetic_queries: list[str]` unconditionally
  (empty list when expansion disabled) for transparency.

Test count 1027 → 1035 (+8 unit tests in
`test_document_expansion.py`).

### 2026-04-25 R4 — Chunk position optimization (Lost-in-the-Middle mitigation)

Liu et al. (TACL 2024, "Lost in the Middle: How Language Models Use Long
Contexts") show that long-context LLMs exhibit U-shaped attention: tokens at
the start and end of the prompt are attended to more than the middle. The
retrieval pipeline emits chunks in score-descending order, which puts the
second-best chunk in the attention-poor middle and wastes the
recency-privileged tail on the lowest-scoring chunk.

`chunks_to_context()` now accepts a keyword-only `ordering` parameter
(`ChunkOrdering` enum) with three variants:

- `SCORE_DESCENDING` (default; current behaviour) — score-descending
  pass-through. Zero behavioural change for existing consumers.
- `PRIMACY_RECENCY` — `evens + reversed(odds)`: best at start, second-best at
  end, weakest in the middle. For input `[c0, c1, c2, c3, c4]` →
  `[c0, c2, c4, c3, c1]`.
- `SANDWICH` — `top_two + reversed(rest)`: top two strongest at the start,
  remainder reversed at the end. For input `[c0, c1, c2, c3, c4]` →
  `[c0, c1, c4, c3, c2]`.

`GenerationConfig.chunk_ordering: ChunkOrdering = ChunkOrdering.SCORE_DESCENDING`
threads the choice through both `GenerationService.generate()` /
`generate_stream()` and `StepGenerationService.generate_step()` uniformly —
no per-call-site knob. Single-chunk and two-chunk inputs are degenerate in
all three orderings (no middle to protect).

This is the start of the R-series (R1–R8 retrieval-quality roadmap items
distinct from Phases A–F). Test count 1018 → 1027 (+4 unit-test functions;
the short-input parametrize expands to 6 cases).

### 2026-04-24 Phase F1 — Domain-bias contract widened to all BAML prompts

The `test_baml_prompt_domain_agnostic` regression guard previously scanned
only two prompt files (`ingestion/functions.baml` + `retrieval/functions.baml`).
It now auto-discovers every `*.baml` file under both `retrieval/baml/baml_src`
and `reasoning/baml/baml_src` (29 files), excluding only `clients.baml` and
`generators.baml` (infrastructure with no prompt bodies) and
`ingestion/drawing.baml` (whose enum-bound `electrical | p_and_id |
mechanical | mixed` strings are `DrawingIngestionConfig`-bound domain
labels rather than prompt examples — the drawing pipeline is inherently
domain-tied via `DrawingIngestionConfig.default_domain`). Zero new
violations were found; future prompts inherit the guardrail automatically.

### 2026-04-24 Phase F2 — DrawingIngestionService status-transition + idempotency tests

Two `@pytest.mark.skip(reason="Pending C4+...")` placeholders in
`test_drawing_service_skeleton.py` were closed out as real e2e
behavioural tests in `test_drawing_e2e_integration.py`:
- `test_phase_methods_reject_status_skip_ahead` — codifies the strict
  4-phase status sequence (`render → extracted → linked → completed`);
  calling `link` on a `'rendered'` source or `ingest` on an `'extracted'`
  source raises `IngestionError`.
- `test_phase_methods_idempotent_on_reentry` — re-entering at a terminal
  status is a no-op; no duplicate vector upserts, no duplicate graph
  writes.

Test count 997 → 999.

### 2026-04-25 Phase F3.5 — Gemini vision provider

One task (1 commit) adding a third arm (`gemini`) to the existing
`Vision` facade alongside the OpenAI and Anthropic implementations.
Consumers now select Gemini the same way they already select an LLM
or embedding provider:
`Vision(LanguageModelProvider(provider="gemini", model=..., api_key=...))`.

Implementation mirrors `_AnthropicVision` / `_OpenAIVision` exactly —
same constructor signature `(provider, max_tokens=4096, max_retries=3)`,
same `async parse(file_path, pages)` contract, same
`MAX_VISION_FILE_SIZE` guard, same unsupported-extension `ValueError`,
same content-policy refusal `ValueError` on empty response, same
`ParsedPage(page_number=1, ..., metadata={"vision_provider": "gemini",
...})` return shape. The only differences are the SDK call
(`client.aio.models.generate_content(model=..., contents=[Part.from_bytes(
data=..., mime_type=...), VISION_EXTRACTION_PROMPT],
config=GenerateContentConfig(max_output_tokens=...))` against
`google.genai`) and the metadata `vision_provider` value.

The `google-genai>=1.0.0` SDK exposes retries via
`HttpOptions(retry_options=HttpRetryOptions(attempts=...))` rather than
as a direct `Client` kwarg, so `max_retries` is wired through there to
preserve facade-level signature parity. The facade's unsupported-provider
error message now lists `gemini` alongside `anthropic` and `openai`.

`MEDIA_TYPES` is unchanged — Gemini accepts the same `image/jpeg`,
`image/png`, `image/gif`, and `image/webp` set the others already
support. Test count 1011 → 1018 (5 unit + 2 facade dispatch).

### 2026-04-25 Phase F3.2 — DXF paperspace layout rendering

One task (1 commit) generalising the DXF render + extract pipeline from
modelspace-only to every layout. `render_dxf` previously rendered only
`doc.modelspace()`; multi-sheet drawings (the mainstream CAD workflow)
silently lost their per-sheet content. The renderer now iterates
`[doc.modelspace()] + [doc.layouts.get(name) for name in
doc.layouts.names_in_taborder() if name.lower() != "model"]` and emits
one splitter-shaped page dict per layout, modelspace at `page_number=1`
and paperspace layouts following in DXF tab order. The renderer's
`RenderContext` is built once per document and reused across layouts;
each layout gets a fresh `matplotlib` figure that is closed
per-iteration to bound the figure cache.

API change: `render_dxf` and `extract_dxf_analysis` now return
`list[...]` instead of a single value. `service.py::render` calls
`render_dxf(...)` directly (was wrapping a single dict in `[ ]`);
`service.py::extract` calls `extract_dxf_analysis(...)` directly (was
wrapping a single `DrawingPageAnalysis` in `[ ]`). Internal-only
refactor — no public SDK consumer touched.

`extract_dxf_analysis` mirrors the page split: each rendered page has a
matching `DrawingPageAnalysis` with `page_number` aligned to the
renderer's numbering. The shared per-layout walk (`_analyse_layout`)
runs the existing `INSERT` -> `DetectedComponent`, `LINE` ->
`DetectedConnection`, and `TEXT`/`MTEXT` -> `OffPageConnector` passes
against `layout.query(...)` instead of `msp.query(...)`. DXF page
dicts now carry `has_images=False` (was `True`) to match the
splitter schema for image-derived flags.

Empty paperspace layouts still render as blank pages — Phase F3.2
favours blank-page-on-empty over silent content loss. No new config
field; paperspace iteration is the universal default. Per-layout DPI
overrides, page-size detection from layout viewports, and invisible-
layout suppression are deferred. Test count 1005 → 1011 (5 unit + 1
e2e).

### 2026-04-26 Phase F3.1 — DXF TEXT/MTEXT off-page-connector detection

One task (1 commit) closing the Phase C deferral that left
`extract_dxf_analysis` emitting `off_page_connectors=[]` even when the
DXF carried explicit cross-sheet tags. The extractor now scans modelspace
`TEXT` + `MTEXT` after the existing `INSERT` + `LINE` pass, regex-matches
each payload against `config.off_page_connector_patterns` (first-match
wins), binds the connector to the underlying component when the text
sits inside a component bbox (reusing `_find_component_at` +
`_CONNECTION_TOL`), and falls through to `bound_component=None` when it
does not. MTEXT formatting codes are stripped via `MText.plain_text()`;
a corrupt MTEXT is skipped with a debug log rather than aborting the
whole sheet. Test count 999 → 1005 (5 unit + 1 e2e).

`OffPageConnector.bound_component` is now `str | None` (was `str`); the
linker passes (`pair_off_page_connectors`, `parse_target_hints`) skip
unbound connectors so a floating sheet annotation can't generate a
`DetectedConnection` with a `None` component id. JSONB round-trip via
`from_dict` switched from `d["bound_component"]` to
`d.get("bound_component")` to honour the new optionality.

### 2026-04-26 Phase E — Remove `QueryAnalysis.domain_hint`

One task (1 commit) deleting the query-side `domain_hint` field entirely.
The field hardcoded an `electrical/mechanical/plc` enumeration in its
BAML prompt (finding from Phase D audit) and its only downstream usage —
`enrich/field_search.py` plumbing it into a `page_type` filter — was
broken by construction after Phase D4 made `page_type` free-form.
Consumers who want domain-aware retrieval already have `source_type` +
`source_type_weights`, `knowledge_id`, and collection partitioning.

Changes:
- `QueryAnalysis` schema no longer has a `domain_hint` field.
- `AnalyzeQuery` prompt no longer enumerates domain categories.
- `StructuredRetrievalService` no longer triggers field-filter search on
  domain_hint and no longer logs it.
- `build_structured_filters` no longer emits a `page_type` filter.
- `test_baml_prompt_domain_agnostic` extended to cover
  `baml_src/retrieval/functions.baml`; scans both ingestion and retrieval
  prompts for domain-bias terms.

`AnalyzeDrawingPage.domain_hint` (ingestion-side, Phase C) is unchanged
— it's a working input parameter, not a broken output field.

### 2026-04-26 Phase D — Agnostic Graph Mapper + Prompt Neutralization

Four tasks (4 feature commits) removing pre-existing electrical/mechanical
domain assumptions from the analyze-path graph mapper and the
`AnalyzePage` / `ExtractEntitiesFromText` BAML prompts. The SDK now
ships zero domain assumption by default; consumers supply their own
vocabulary via `GraphIngestionConfig`. Test count 986 → 997.

**D1 — `GraphIngestionConfig`.** New nested config on `IngestionConfig.graph`
with three knobs: `entity_type_patterns` (list of `(regex, type_name)`
pairs, first match wins), `relationship_keyword_map` (keyword → Neo4j
relation_type; every target validated against `ALLOWED_RELATION_TYPES` at
`__post_init__`), and `unclassified_relation_default` (fallback edge type
when no keyword matches; default `"MENTIONS"`, set to `None` for strict
drop). Registered with `test_config_bounds_contract`.
Commit: `4639aff`.

**D2 — Mapper refactor.** `page_entities_to_graph` and
`cross_refs_to_graph_relations` now accept a `GraphIngestionConfig`
argument. Removed `_ELECTRICAL_PATTERNS`, `_MECHANICAL_PATTERNS`, and
`_RELATIONSHIP_KEYWORDS` module-level constants. Entity-type inference
falls back to `category.lower()` → `"entity"` when config patterns don't
match. Cross-reference classification falls back to
`unclassified_relation_default` instead of silently dropping — the
pre-Phase-D silent drop was a correctness bug for any non-electrical
domain. Tests rewritten to exercise the config mechanism.
Commit: `5522c85`.

**D3 — Wiring.** `AnalyzedIngestionService` and `GraphIngestion` both
accept `graph_config: GraphIngestionConfig | None`. `RagEngine` passes
`IngestionConfig.graph` through. When unset, services construct a
default empty-vocab + MENTIONS-fallback config, so existing non-drawing
flows become agnostic by default.
Commit: `5d7562f`.

**D4 — Neutralized BAML prompts.** `AnalyzePage` and
`ExtractEntitiesFromText` prompt bodies stripped of "valves, motors,
wires, terminals" / "480V, 175 PSI, 3450 RPM" / "Motor M1" / "SAE" /
"RV-2201" / "electrical" / "mechanical" bias terms. Replaced with
structure-focused instructions that describe the expected shape
(entities, categories, tables) without seeding the LLM with domain
vocabulary. New contract test prevents regression.
Commit: `f7ffdfc`.

### 2026-04-25 Phase C — DrawingIngestion MVP

Thirteen tasks (13 feature + 2 fix-up commits, plus docs) shipping a first-class SDK
capability for ingesting technical drawings (schematics, P&ID, wiring,
mechanical) with multi-page connectivity resolution. Input formats: PDF
(vision-based) and DXF (structured parse via ezdxf). All drawing standards,
symbol vocabularies, off-page-connector patterns, and relation-type
vocabularies are consumer-configurable (SDK ships IEC 60617 + ISA 5.1
defaults). Test count 890 → 986. ruff + mypy clean on 345+ source files.

Architecture: new `DrawingIngestionService` sibling of `AnalyzedIngestionService`
with a 4-phase pipeline: `render → extract → link → ingest`. The `link`
phase is the novel piece — deterministic cross-sheet resolution first
(exact off-page tag matching, regex target hints, RapidFuzz label merges),
LLM synthesis only for the residue. Graph mapping reuses the existing Neo4j
allowlist (`CONNECTS_TO` / `FEEDS` / `FLOWS_TO` / `POWERED_BY` /
`CONTROLLED_BY` / `MENTIONS` / `REFERENCES`) — no schema migration.

**C1 — BAML schema + functions.**
`baml_src/ingestion/drawing.baml`: new classes (`Port`,
`DetectedComponent`, `DetectedConnection`, `OffPageConnector`,
`DrawingPageAnalysis`, `DrawingSetSynthesis`, `Merge`, `NarrativeXref`) and
two functions (`AnalyzeDrawingPage`, `SynthesizeDrawingSet`). All
user-controlled string parameters fenced per the prompt-fence contract.
Commit: `c665618`.

**C2 — `DrawingIngestionConfig`.**
Nested into `IngestionConfig.drawings` (matches `tree_search` precedent).
Consumer-overridable: `symbol_library` (full replace or
`symbol_library_extensions` additive), `off_page_connector_patterns`
(regex list, default covers `/A2`, `OPC-N`, `to sheet N zone XN` idioms),
`relation_vocabulary` (wire_style → relation_type; `__post_init__`
validates every target against `ALLOWED_RELATION_TYPES`),
`fuzzy_label_threshold`, `analyze_concurrency`, `dpi`,
`graph_write_batch_size` — all bounded per the config-bounds contract.
Commits: `00ae462`, `06c651f`.

**C3 — Service skeleton + dataclass models.**
`DrawingIngestionService` sibling of `AnalyzedIngestionService` with 4
async phase stubs. Python dataclasses (`Port`, `DetectedComponent`,
`DetectedConnection`, `OffPageConnector`, `DrawingPageAnalysis`) mirror
the BAML schema with `to_dict`/`from_dict` for JSONB persistence.
Commit: `bd7d2da`.

**C4 — Render phase.**
PDF via PyMuPDF (reused `iter_pdf_page_images`), DXF via
`ezdxf.addons.drawing.matplotlib` rendered to single-sheet PNG. File-hash
idempotency short-circuit. New deps: `ezdxf>=1.3,<2.0` and
`matplotlib>=3.8,<4.0`. Matplotlib forced to Agg backend for
asyncio-worker-thread safety.
Commit: `a6fa302`.

**C5 — Extract phase (PDF).**
Per-page `AnalyzeDrawingPage` BAML call with consumer-supplied
`symbol_library` + `off_page_patterns`. Semaphore-capped at
`config.analyze_concurrency`. Merges LLM-extracted analysis into the
existing `rag_page_analyses.data` row preserving `page_image_b64` from
render. Idempotent on re-entry when status already `extracted`.
Commit: `dde63cb`.

**C6 — Extract phase (DXF).**
INSERT entities → components via exact-then-substring classification against
`symbol_library`. LINE entities whose endpoints fall inside two distinct
bboxes (with small tolerance) → connections. Zero LLM calls. Off-page
connectors in TEXT/MTEXT deferred to Phase D.
Commits: `4291b39`, `ec56036` (fix: tighten proximity tolerance to a
2-unit absolute constant after spec review flagged the proportional
version as production risk).

**C7 — Link phase (deterministic).**
Three passes, all deterministic: (1) `pair_off_page_connectors`
exact-tag match produces consecutive DetectedConnection pairings; (2)
`parse_target_hints` regex `sheet N (zone XN)?` resolves target pages;
(3) `merge_fuzzy_labels` RapidFuzz WRatio above
`fuzzy_label_threshold`, each component consumed once. New dep:
`rapidfuzz>=3.5,<4.0`.
Commit: `d83488f`.

**C8 — Link phase (LLM residue).**
`SynthesizeDrawingSet` only called when `multi_page_linking=True`, an
`lm_client` is configured, AND unresolved candidates remain. Merges
below 0.5 confidence dropped. BAML errors logged and swallowed —
deterministic pairings still commit.
Commit: `b3cf2bc`.

**C9 — Drawing-specific graph mapper.**
Threads bbox/ports/domain/page_number into `GraphEntity.properties`;
encodes wire_style/net/from_port/to_port/cross_sheet into a deterministic
`k=v;k=v` `GraphRelation.context` string (GraphRelation has no
`properties` dict in the shared dataclass). Drops the analyze path's
`len(shared_entities)>=2` filter that kills legitimate single-component
cross-sheet wires. LLM-suggested merges emit `MENTIONS` edges with
`llm_suggested=true` in the context.
Commit: `ece88af`.

**C10 — Ingest phase.**
One vector per component (`vector_role='drawing_component'`,
`source_type='drawing'`, embedding-friendly description = symbol_class +
label + sorted same-page neighbours + domain). Graph writes batched by
`graph_write_batch_size` (default 500) to avoid large Neo4j
transactions. Idempotent on `completed`. No-op graph path when
`graph_store is None`.
Commit: `dfd588a`.

**C11 — `RagEngine` routing.**
`.dxf` always routes to the drawing service; `.pdf` routes only when
`source_type="drawing"` (tiebreaker). `SUPPORTED_STRUCTURED_EXTENSIONS`
stays `{.xml, .l5x}` — no .pdf regression. Stepped API:
`render_drawing`, `extract_drawing`, `link_drawing`,
`complete_drawing_ingestion`. Status-based resume through all four
phases. `collection=` arg rejected. `shutdown()` drops the service for
lifecycle hygiene.
Commit: `df05d92`.

**C12 — `GraphRetrieval.trace()` helper.**
Thin wrapper around the graph store's `query_graph` + N-hop traversal,
returning `GraphPath` objects directly (vs `search()`'s RetrievedChunk
conversion). `relation_types` is a strict AND filter. Store errors
convert to an empty list + warning log. Enables spatial queries like
"what feeds V-101" on drawing-ingested knowledge.
Commit: `cd8d22e`.

**C13 — End-to-end integration + docs.**
Real ezdxf round-trip fixture (`simple_rlc.dxf`) through all four phases
with stubbed stores. Consumer `symbol_library` override test. File-hash
idempotency test.
Commit: `710516a`.

### 2026-04-25 Phase B — Perf/Cost Prerequisites

Six tasks (7 commits: 6 feature + 1 scope-fix) reducing LLM call cost and
enabling safe resumption of interrupted ingestion runs.
Test count 861 → 890. ruff + mypy clean on 318+ source files.

**B1 — `rag_page_analyses` table.**
New `(source_id, page_number)` composite-key table with indexed `page_hash`
column. `upsert_page_analyses` + `get_page_analyses` + `get_page_analysis` on
`SQLAlchemyMetadataStore` with dialect-dispatched upsert (SQLite + Postgres).
Idempotent migration via `_migrate_missing_columns`. FK cascade-delete from
`rag_sources.id`.
Commit: `1a3ceb2`.

**B2 — Migrate AnalyzedIngestionService.**
All 3 phases (analyze / synthesize / ingest) and all 3 file formats (PDF / L5X
/ XML) now read/write `PageAnalysis` blobs via the dedicated table instead of
serialising into `source.metadata["page_analyses"]`. `synthesis` blob stays in
`source.metadata`.
Commit: `2bf9a5d`.

**B3 — File-hash + per-page-hash caching.**
Two caching layers: (a) `analyze()` short-circuits on
`find_by_hash(file_hash, knowledge_id)` when status ∈ {analyzed, synthesized,
completed}; (b) per-page SHA-256 cache: each rendered page's PNG bytes are
hashed, and `get_page_analyses_by_hash(page_hashes, knowledge_id=None)`
resolves cache hits in bulk. A 1-page revision of a 500-page manual now fires
1 LLM call instead of 500. `PageAnalysis.page_hash: str` added.
Commit: `7e189b1`.

**B4 — Status-based resume + phase idempotency.**
`RagEngine.ingest()`'s structured branch now inspects `Source.status` and
routes to the first unfinished phase (completed → no-op; synthesized → ingest;
analyzed → synthesize+ingest; otherwise → full 3-phase run). `synthesize()` and
`ingest()` phase methods are idempotent on re-entry. **Fix-up commit** reverted
an out-of-scope `.pdf` routing change that would have silently routed every PDF
through the expensive analyzed pipeline.
Commits: `c05148f`, `ad539a9`.

**B5 — PyMuPDF text-density pre-filter.**
Pages with `>= analyze_text_skip_threshold_chars` extractable text AND zero
embedded images are classified as `page_type="text"` and built from raw text
directly — no vision LLM call. Typical 30–50% saving on narrative-heavy
engineering manuals at zero new deps. Threshold bounded `[0, 100_000]`; default
300; 0 disables.
Commit: `5dc20b8`.

**B6 — `analyze_concurrency` config.**
Promoted `_ANALYZE_PDF_CONCURRENCY=5` module constant to
`IngestionConfig.analyze_concurrency: int = 5` with bounds `[1, 100]`. Tier 2+
API accounts can safely crank higher.
Commit: `b4291fb`.

### 2026-04-25 Phase A — RAG Quality Baseline

Five RAG-quality tasks (9 commits: 5 feature + 4 follow-up) targeting
retrieval fidelity at the chunking, ingestion, and retrieval layers.
Test count 837 → 861. ruff + mypy clean on 317 source files.

**A1 — Token-aware chunk sizing.**
`SemanticChunker` and `IngestionConfig` accept `chunk_size_unit: Literal["chars","tokens"]`
defaulting to `"tokens"`. Uses `tiktoken.get_encoding("cl100k_base")` with a
word-count fallback when tiktoken is unavailable. Default `chunk_size` changed
from 500 chars to **375 tokens**; `chunk_overlap` from 50 to **40**. New
dependency `tiktoken>=0.5,<1.0`. New module `token_counter.py`.
Commits: `5e7a37e`, `8d1a32b`.

**A2 — Hard-split guard.**
`RecursiveTextSplitter` now forces deterministic boundary slicing when the
separator ladder reaches `""` or a piece exceeds `chunk_size` with no
remaining separators. Previously emitted raw oversized chunks that embedding
providers silently truncated. `ChunkedContent.was_hard_split: bool` surfaces
the flag for generation provenance. New public
`split_text_with_flags() -> list[tuple[str, bool]]`.
Commits: `ccaaab8`, `3a4f7b8`.

**A3 — Structure-aware chunking.**
New `structure.py` with `find_atomic_regions`, `build_heading_spans`,
`section_path_at`. Markdown tables and fenced code blocks are now atomic chunks
(unless they exceed `chunk_size`). Markdown heading hierarchy populates
`ChunkedContent.section` (e.g. `"Safety > Lockout procedures > Step 2"`).
Closes a long-standing RAG failure where table headers and row values landed in
separate chunks. Excludes heading-like lines inside code fences; strips
CommonMark ATX closing `#` sequences. Parent-child path gets the same
atomic-aware treatment.
Commits: `189d363`, `ead6225`.

**A4 — Multi-vector analyzed ingest.**
`AnalyzedIngestionService.ingest` now produces 3 vector kinds per page:
`description` (LLM prose enriched for embedding), `raw_text` (PyMuPDF
`page.get_text()` output, conditional on non-empty), `table_row` (one vector
per row per table, column-header-prefixed). Each payload tagged by
`vector_role`. `full_text` passed to downstream document-store methods is now
raw OCR text (fallback: LLM description with entities for L5X/XML).
`PageAnalysis.raw_text: str` added with back-compat deserialization.
Commits: `372548d`, `537906c`.

**A5 — Parent-child on by default + score aggregation.**
`IngestionConfig.parent_chunk_size` default flipped `0 → -1` sentinel,
resolved to `3 * chunk_size` in `__post_init__`. Explicit
`parent_chunk_size=0` still disables. `VectorRetrieval._expand_parents` now
delegates to a new `_merge_children_into_parents` helper that groups children
by `parent_id`, **sums child scores per parent**, and records
`child_hit_count` / `expanded_from_children` in payload. Captures the
multi-hit strength that was silently collapsed.
Commit: `3078c18`.

**Deferred to future round-6 (not in Phase A):**
- Finding 6 — `chunk_context_headers` pollutes embeddings with filename/page-number.
- Finding 7 — `len/4` token heuristic in tree indexing.
- Finding 8 — RRF `k=60` hardcoded + flat weights + tree chunks under-weighted.
- Finding 9 — Grounding gate `max(score)` ambiguous across score scales.
- Finding 10 — BM25 tokeniser strips hyphens.

### 2026-04-24 Round 4 Review

Cumulative fixes from the fourth-pass SDK audit. Primary achievement:
**four systematic contract-test sweeps** that now block future instances of
the same findings. 21 commits; test count 789 → 837.

**Contract-test sweeps (prevent future circling):**
- `test_baml_prompt_fence_contract.py` — every user-controlled BAML prompt parameter across retrieval + reasoning must be fenced. Sweep caught and fixed **35 unfenced parameters** across 24 functions in 13 BAML files (rounds 2-3 had only spot-fixed 7).
- `test_config_bounds_contract.py` — every int/float config field must have validation in `__post_init__` or a `# unbounded: <reason>` marker. Sweep fixed **8 unbounded knobs** (source_type_weights, temperature, num_variants, ClassificationConfig.concurrency, ClusteringConfig.{n_clusters,min_cluster_size,random_state}, ComplianceConfig.max_reference_length).
- Serial-await audit — 11 intentionally-serial loops annotated with `# SERIAL:` comments explaining why they aren't parallelised.
- `test_no_bare_valueerror_in_configs.py` — 33 bare `raise ValueError` replaced with new `ReasoningInputError(ReasoningError, ValueError)` across 7 reasoning modules.

**Ship blockers fixed:**
- `TreeRetrieval` dead code removed (never wired into MethodNamespace; round-3 parallelised an unreachable path) (A1).
- Lucene metacharacter escape on Neo4j fulltext queries — prevents wildcard-explosion DoS (A2/H1).
- `AnalyzedIngestionService._analyze_pdf` parallelised with bounded `asyncio.gather` (5 concurrent) — 200-page PDF no longer 200 serial LLM calls (A3/H2).
- `parent_chunk_overlap < parent_chunk_size` validator added — prevents infinite `_merge_splits` loop (A4/H4).
- `_migrate_missing_columns` verified idempotent via column introspection (A5/M1).
- Neo4j URI credentials scrubbed before startup logging (A6/M5).

**Performance:**
- Batch tree-index lookup eliminates N+1 DB query in `_run_tree_search` (C1/M6).
- `list_source_ids` lean projection — tree search no longer transfers megabyte `metadata_json`/`tree_index_json` payloads (C2/M7).
- `embed_batched` centralises sub-batch concurrency with `asyncio.Semaphore(3)`; removes competing provider-layer gather (C3/M8).
- BAML `ClientRegistry` cached in `GenerationService.__init__` instead of rebuilt per `generate()` call (D1/L7).
- `_synthesize_shared_entities` caps per-entity pairwise cross-reference expansion at 20 pages (D6).

**Security:**
- `TreeIndex.pages` count guard (≤ 100_000) added to `from_dict`, complementing round-3's depth/node guards on `TreeNode` (C4/F7).
- `compliance --references` CLI: `exists=True` + symlink/traversal containment check in `_read_directory_as_text` (D3/F5).
- Upper-bound pins on `lxml`, `fastembed`, `voyageai`, `neo4j`, `rank-bm25` (D4/F6).

**Safety / lifecycle:**
- `RagEngine.shutdown` idempotent (separate `_stores_opened` flag preserves the init-rollback path) (D5/L2).

**Docs:**
- CLAUDE.md test count, new config ceilings, collection-0 aliasing note (D2 / D7 / arch M10).

**Deferred from prior rounds:**
- Round-3 L8 (uniform scoped pipelines / remove index-0 special case) — comment extended (D7) but refactor still not pursued; tests codify the current behavior.

### 2026-04-24 Round 3 Review

Cumulative fixes from a third-pass SDK audit. 25 commits; test count 741 → 789.

**Round 2 regressions resolved:**
- `on_progress` callback now emits a final `(total, total)` completion event regardless of group composition (R2-A).
- `_dispatch_methods` TaskGroup `except*` handler now propagates `CancelledError`/`SystemExit`/`KeyboardInterrupt` unwrapped (R2-B).

**Performance:**
- BM25 index build scrolls chunks outside the cache lock — concurrent builds for different `knowledge_id`s no longer serialize (P-1).
- `TreeRetrieval.search` fans out per-source work with `asyncio.gather` (P-2).
- Embedding provider sub-batches run under a bounded `asyncio.Semaphore(3)` instead of serial `for` loop (P-3).
- Added DB index on `rag_sources.file_hash` with schema migration — `find_by_hash` no longer full-scans (P-4).
- `VectorIngestion` gathers dense + sparse embeddings concurrently (P-5).

**Config safety:**
- `RFNRY_RAG_BAML_LOG` validated against `{trace, debug, info, warn, error, off}` (C-1).
- All stores reject zero/negative timeout and pool knobs; SQLAlchemy accepts `-1` for `pool_recycle` (C-2).
- `TreeSearchConfig.max_sources_per_query` (`1–1000`, default 50) caps `_run_tree_search` fan-out (C-3).
- Upper bounds on tree indexing/search knobs: `toc_scan_pages ≤ 500`, `max_pages_per_node ≤ 200`, `max_tokens_per_node ≤ 200_000`, `max_steps ≤ 50`, `max_context_tokens ≤ 500_000` (C-4).
- `BatchConfig.batch_size ≤ 100_000` (C-5).
- `RetrievalConfig.bm25_max_indexes ∈ [1, 1000]` (L2).

**Security hardening:**
- Query-rewriting BAML prompts (HyDE, multi-query, step-back) fence `{{ query }}` in `======== QUERY START/END ========` (S-1).
- Reasoning BAML `AnalyzeContext` fences `{{ roles }}`; `CheckCompliance` fences `{{ dimensions }}` (S-2).
- Reasoning SDK configs cap `max_text_length ≤ 5_000_000` chars (S-3).
- `RagEngine.embed` and `embed_single` validate query length (L3).
- `TreeNode.from_dict` rejects tampered trees with depth > 100 or > 10 000 nodes (L4).
- `mapper._classify_relationship` returns `None` for unknown relationships instead of defaulting to `CONNECTS_TO`; unclassifiable cross-references are dropped (L5).

**Error typing:**
- `RuntimeError` → `ConfigurationError` for missing generation config at call time (three sites) (L6).
- New `InputError(RagError, ValueError)` for input-validation failures; back-compat with `except ValueError:` preserved (L7).

**Observability:**
- Neo4j and Qdrant stores log effective pool/timeout settings at startup (L1).

**Docs:**
- Clarified `RetrievalService.methods` is a concrete-class contract (L9).
- CLAUDE.md documents tree-search placement in the retrieval pipeline flow (L10).
- `Pipeline` docstring explains why `ClusteringService` is intentionally excluded (L11).

**Deferred:**
- L8 (uniform scoped pipelines) — deferred. The `collections[0]` special case is intentional per existing tests that codify first-collection-reuses-default-services behavior. Will require test-spec revision before it can land.

### 2026-04-23 Round 2 Review

Cumulative fixes from a second-pass SDK audit. 40 commits; test count 689 → 741.

**Correctness (critical & high):**
- Rejected `collection=` on structured ingestion paths instead of silently using the default collection (C1).
- Wired the previously-noop `on_progress` callback in `RagEngine.ingest` (C2).
- Fan out `_run_tree_search` across sources with `asyncio.gather` — serial loop removed (H1).
- Hardened Cypher relationship-type allowlist with a contract test and fixed a latent silent-fallback bug in `_validate_relation_type` (H2).
- Replaced private `_retrieval_methods` reach-through with a public `RetrievalService.methods` property (H3).
- Selected `DocumentIngestion` by `isinstance` rather than string-name filter (H4).

**Correctness (medium):**
- Scoped-collection ingestion now includes graph and tree methods (M1).
- Analyzed ingestion writes the document store exactly once in phase 3 (M2).
- `grounding_enabled` guard moved to `GenerationConfig.__post_init__` (M3).
- `tree_indexing.enabled` and `tree_search.enabled` without a `model` now fail at init (M4).
- Ingestion methods parallelise within required/optional groups via TaskGroup + gather (M5).

**Performance:**
- Voyage reranker uses `voyageai.AsyncClient` directly (M6).
- Embedding providers chunk batches at provider limits: OpenAI 2048, Voyage 128, Cohere 96 (M7).
- Metadata-store duplicate-hash check pushed into SQL WHERE (L12).

**Security hardening:**
- `GenerateAnswer` BAML prompt fences untrusted context to blunt prompt injection (M8).
- Postgres document search moved from f-string SQL assembly to SQLAlchemy Core expressions (M9).
- PyMuPDF upper-bounded; PDF parser guards size and page count (M10).
- `RagEngine.generate_step` validates query length (L13).
- Filesystem frontmatter values JSON-encoded to survive embedded delimiters (L14).
- Reasoning CLI directory reads capped at 5 MB aggregate (L15).

**Configuration surface:**
- Added `QdrantVectorStore(hybrid_prefetch_multiplier=...)` (L4).
- Added `PostgresDocumentStore(headline_max_words, headline_min_words, headline_max_fragments)` (L5).
- Added `RetrievalConfig(history_window=...)` (L6).
- `RFNRY_RAG_LOG_LEVEL` validates against known level names (L7).

**Observability:**
- Effective pool sizes and LLM client policy logged at startup (L8).

**Lifecycle / safety:**
- `run_concurrent` caller-supplied `concurrency` bounded to 1–100 (L9).
- `RagEngine.shutdown` tears down stores in reverse-init order and clears service refs (L10).
- BM25 cache initial read held under lock to prevent torn read vs invalidate (L11).

**Refactors & hygiene:**
- `IngestionService` internal kwarg renamed `contextual_chunking` → `chunk_context_headers` (L1).
- `MethodNamespace[T]` type parameter threaded through engine properties (L2).
- Shared synthesise-entities loop body extracted in analyzed ingestion (L3).
- `_escalation_result` loses unused `chunks=` parameter (M11).
- `_enabled_flows()` normalised to snake_case (M12).
- `reasoning/__init__.py::__all__` sorted; `RUF022` enabled (M13).
- `generate_step()` documented in retrieval README (M14).

### Breaking changes

- **`rfnry_rag.common.errors.BaseException` renamed to `SdkBaseError`**, and
  no longer re-exported at the top level of `rfnry_rag`. The old name
  shadowed the Python builtin: any code doing `from rfnry_rag import *`
  or `from rfnry_rag import BaseException` silently narrowed user
  `except BaseException:` clauses. Users should catch the specific
  subclasses (`RagError`, `ReasoningError`, `ConfigurationError`) or
  import `SdkBaseError` from `rfnry_rag.common.errors` directly.
- **`Neo4jGraphStore(password=...)` is now required.** The previous default
  `"password"` was the Neo4j community-edition default — the universally
  known credential was a landmine. Passing an empty password raises
  `ConfigurationError`. Neo4j pool defaults also tightened:
  `max_connection_pool_size=10` (was 100), `connection_acquisition_timeout=5.0s`
  (was 60.0s).
- **`IngestionConfig.contextual_chunking` deprecated; use
  `IngestionConfig.chunk_context_headers`.** The old name implied
  Anthropic-style LLM-generated per-chunk context; the implementation is
  pure string templating. The old name still works for one release but
  emits `DeprecationWarning`. CLI TOML: prefer `chunk_context_headers = true`.
- **`BOUNDARY_API_KEY` collision raises `ConfigurationError`** instead of
  warning. If two `LanguageModelClient` instances register different keys
  in the same process, the second raises — set `BOUNDARY_API_KEY` once at
  process start in multi-tenant scenarios.

### Added

- `LanguageModelClient.timeout_seconds` (default `60s`) — per-LLM-call
  timeout. Previously no timeout; a hung call could stall the event loop.
- `QdrantVectorStore.scroll_timeout` (default `30s`),
  `QdrantVectorStore.write_timeout` (default `30s`),
  `QdrantVectorStore.max_scroll_limit` (default `10_000`) — per-operation
  timeouts and scroll cap. Plaintext `http://` URL without an API key now
  emits a startup warning.
- `SQLAlchemyMetadataStore.pool_timeout` and
  `PostgresDocumentStore.pool_timeout` (default `10s`, was SQLAlchemy's
  silent 30s) — fail-fast on pool exhaustion.
- `Neo4jGraphStore.connection_timeout` (default `5.0s`) — explicit TCP
  connect timeout.
- `BaseIngestionMethod.required: bool` — vector/document default `True`,
  graph/tree default `False`. Failures in required methods raise
  `IngestionError` and skip the metadata commit so a source is never
  marked successful without its vectors.
- `rag_schema_meta` version table on `SQLAlchemyMetadataStore` — migrations
  are idempotent under concurrent process starts and refuse to operate
  against a schema newer than the code (downgrade protection).
- `ValueError` on `MethodNamespace` duplicate method names.
- `retrieval/common/concurrency.py` — re-export of `run_concurrent` for
  parity with `reasoning/common/concurrency.py`.
- `BaseChunkRefinement`, `BaseRetrievalJudgment` exported from
  `rfnry_rag.retrieval`; `BaseEmbeddings`, `BaseSemanticIndex` exported
  from `rfnry_rag.reasoning`.
- `rfnry_rag.common.embeddings.embed_batched` — shared between retrieval
  ingestion and reasoning clustering.
- `rfnry_rag.common.cli.OutputMode`, `get_output_mode`, `get_api_key` —
  shared CLI helpers (previously duplicated across retrieval/reasoning).
- `rfnry_rag.common.logging.query_logging_enabled` — gate helper for
  `RFNRY_RAG_LOG_QUERIES`.

### Changed

- `_merge_retrieval_results` now RRF-merges structured + unstructured
  results instead of raw-score sort. Previously structured cosine
  scores (0–1) always beat unstructured RRF scores (0.01–0.05)
  regardless of relevance.
- `RagEngine.initialize()` wraps setup in try/except and calls
  `shutdown()` on partial failure. Previously, an exception in
  mid-init leaked prior store connections (`__aexit__` doesn't fire
  when `__aenter__` raises).
- Multi-collection wiring: `_retrieval_by_collection` and
  `_ingestion_by_collection` populated symmetrically at `initialize()`.
  Unknown collection names raise `ValueError` instead of silently
  falling back to the default pipeline.
- `_on_source_removed` / `_on_ingestion_complete` fan out BM25-cache
  invalidation across every scoped collection, not just the default.
- `AnalyzedIngestionService` hashes files via `asyncio.to_thread`
  (previously blocked the event loop).
- `BatchIngestionService` drain uses `return_exceptions=True` so a
  single failing task doesn't cancel peers.
- `_retrieve_chunks` uses `return_exceptions=True`; one failing
  retrieval path logs + continues instead of killing the query.
- `search/service.retrieve()` raises `RetrievalError` when every query
  variant fails (was silently returning `[]`).
- `generate_stream` emits a terminal `StreamEvent(type="done",
  grounded=False)` before re-raising `GenerationError`.
- Neo4j `delete_by_source` uses `async with session.begin_transaction()`
  for guaranteed cleanup on commit-path server errors.
- BAML `Default` fallback client model set to sentinel
  `"UNCONFIGURED-route-through-build-registry"` so unrouted calls fail
  fast with a clear error instead of silently using `gpt-4o-mini`.

### Security

- **XXE hardening**: XML and L5X parsers use an explicit
  `etree.XMLParser(resolve_entities=False, no_network=True,
  load_dtd=False, huge_tree=False)` on every read.
- **Path-traversal rejection**: `FilesystemDocumentStore` validates
  `knowledge_id` and `source_type` against
  `^[A-Za-z0-9_-][A-Za-z0-9_-.]{0,127}$`, asserts resolved-path
  containment under `base_path`, and skips symlinks during `rglob`
  traversal.
- **Credential redaction**: `LanguageModelProvider.api_key`,
  `Neo4jGraphStore.password`/`username`, and
  `LanguageModelClient.boundary_api_key` marked `repr=False` so they
  never appear in tracebacks or log lines.
- **Query log gate**: the `step_back` rewriter previously logged raw
  query text at INFO bypassing the privacy gate. Now consistently
  gated behind `RFNRY_RAG_LOG_QUERIES=true` via the shared
  `query_logging_enabled()` helper.
- **`reasoning init` chmod 0o600** on generated `config.toml`, matching
  retrieval `init`.
- **Input size limits** at public entry points: query ≤ 32 000 chars,
  `ingest_text` content ≤ 5 000 000 chars, metadata ≤ 50 keys × 8 000
  chars.

### Fixed

- `RagEngine.vector_only(...)`, `.document_only(...)`, and `.hybrid(...)`
  presets now `initialize()` cleanly. Previously the grounding-gate
  guard fired whenever `grounding_threshold > 0` (default `0.5`)
  regardless of whether grounding was enabled, raising
  `ConfigurationError` on every retrieval-only preset.
- Required ingestion methods no longer silently commit a source when
  their upsert fails (vector upsert failure previously resulted in a
  valid-looking source with zero vectors in Qdrant).
- `check_embedding_migration` early-returns when no embedding model is
  configured so retrieval-only/document-only setups don't flood logs
  with stale-source warnings.

### Validation tightened

- `IngestionConfig.dpi`: `72 ≤ dpi ≤ 600`.
- `RetrievalConfig.top_k`: `≤ 200`.
- `RetrievalConfig.bm25_max_chunks`: `≤ 200 000`.
- `GenerationConfig`: reject `grounding_enabled=True` with
  `grounding_threshold=0.0` (no-op).
- Cross-config: `tree_indexing.max_tokens_per_node ≤
  tree_search.max_context_tokens`.
- `BatchConfig.concurrency`: cap at `20`.
- Reasoning CLI TOML: unknown top-level keys now raise `ConfigError`
  (matches retrieval's prior hardening).

### Removed

- `retrieval/modules/retrieval/search/vector.py` (`VectorSearch` class —
  orphaned; logic duplicated inside `methods/vector.py`).
- Stale `# src/rfnry-rag/retrieval/modules/retrieval/base.py` banner
  comment (referenced the old hyphenated package path).

---

## Prior history

See `git log --oneline` for commits predating the 2026-04-23 review.
