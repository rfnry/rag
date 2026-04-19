# SDK Cleanup — Design

**Date:** 2026-04-10
**Scope:** Normalize naming, patterns, and public API surface across both SDKs (retrieval + reasoning). Pre-1.0, breaking changes allowed.

## Goals

1. One consistent pattern for provider-backed clients (embeddings, vision, rerankers, LLM calls).
2. Clear, non-overlapping env var surface with no implicit vendor-key leaks.
3. Consistent naming for pipeline components (no mix of agent-nouns and gerunds).
4. Strict, minimal public API surface in `__init__.py` exports.
5. Remove legacy cryptic names (`AceError`) and misleading docstrings (`RAC`).
6. Deduplicate Protocols defined in multiple places with divergent contracts.

## Non-Goals

- Functional refactoring beyond rename-and-consolidate. No new features.
- Plugin-author API (deferred; `Base*` protocols go private for now).
- Performance changes.

---

## 1. Provider primitive and active client renames

### 1.1 `LanguageModelProvider` — shared identity triple

A new shared primitive in `x64rag/common/language_model.py`:

```python
@dataclass
class LanguageModelProvider:
    """Identity: which API, which model, auth key.
    Used directly by Embeddings/Vision/Reranking, or wrapped by LanguageModelClient."""
    provider: str           # "openai" | "anthropic" | "cohere" | "voyage" | ...
    model: str
    api_key: str | None = None
```

Replaces `LanguageModelClientConfig` everywhere.

### 1.2 `LanguageModelClient` — BAML wrapper

```python
@dataclass
class LanguageModelClient:
    """BAML-backed LLM client with routing, retries, fallback, and generation params."""
    provider: LanguageModelProvider
    fallback: LanguageModelProvider | None = None
    max_retries: int = 3
    strategy: Literal["primary_only", "fallback"] = "primary_only"
    max_tokens: int = 4096       # LLM-only, moved off the primitive
    temperature: float = 0.0     # LLM-only, moved off the primitive
    boundary_api_key: str | None = None
```

Replaces `LanguageModelConfig`. `max_tokens` and `temperature` migrate off the primitive because they don't apply to embeddings/reranking.

### 1.3 Active client facades — dispatch internally

Provider-specific concrete classes collapse into three facades. Old concrete classes become private.

**Embeddings**
```python
class Embeddings:
    def __init__(self, provider: LanguageModelProvider) -> None:
        match provider.provider:
            case "openai":  self._impl = _OpenAIEmbeddings(provider)
            case "voyage":  self._impl = _VoyageEmbeddings(provider)
            case "cohere":  self._impl = _CohereEmbeddings(provider)
            case _: raise ConfigurationError(f"Unsupported embeddings provider: {provider.provider!r}")
    async def embed(self, texts): return await self._impl.embed(texts)
    async def embedding_dimension(self): return await self._impl.embedding_dimension()
```

**Vision**
```python
class Vision:
    def __init__(self, provider: LanguageModelProvider) -> None:
        match provider.provider:
            case "anthropic": self._impl = _AnthropicVision(provider)
            case "openai":    self._impl = _OpenAIVision(provider)
            case _: raise ConfigurationError(...)
```

**Reranking** — accepts either primitive, dispatching on type
```python
class Reranking:
    def __init__(self, config: LanguageModelProvider | LanguageModelClient) -> None:
        if isinstance(config, LanguageModelClient):
            self._impl = _LLMReranking(config)        # BAML path — any LLM
        else:
            if config.provider not in {"cohere", "voyage"}:
                raise ConfigurationError(
                    f"Provider {config.provider!r} has no dedicated reranker API. "
                    f"Wrap it in LanguageModelClient to use LLM-as-reranker."
                )
            self._impl = _DedicatedReranking(config)  # direct SDK path
```

The input **type** selects the mechanism: `LanguageModelProvider` → dedicated API (Cohere/Voyage), `LanguageModelClient` → LLM-as-reranker via BAML. One class, two cost/quality profiles, no class explosion.

### 1.4 Class rename table (complete)

| Before | After | File(s) |
|---|---|---|
| `LanguageModelClientConfig` | `LanguageModelProvider` | `common/language_model.py` |
| `LanguageModelConfig` | `LanguageModelClient` | `common/language_model.py` |
| `OpenAIEmbeddings`, `VoyageEmbeddings`, `CohereEmbeddings` | `Embeddings` (facade) + private `_OpenAIEmbeddings`, `_VoyageEmbeddings`, `_CohereEmbeddings` | `retrieval/modules/ingestion/embeddings/` |
| `AnthropicVision`, `OpenAIVision` | `Vision` (facade) + private `_AnthropicVision`, `_OpenAIVision` | `retrieval/modules/ingestion/vision/` |
| `CohereReranking`, `VoyageReranking`, `LLMReranking` | `Reranking` (unified facade) + private `_CohereReranking`, `_VoyageReranking`, `_LLMReranking` | `retrieval/modules/retrieval/search/reranking/` |
| `ExtractiveRefiner` | `ExtractiveRefinement` | `retrieval/modules/retrieval/refinement/extractive.py` |
| `AbstractiveRefiner` | `AbstractiveRefinement` | `retrieval/modules/retrieval/refinement/abstractive.py` |
| `BaseChunkRefiner` | `BaseChunkRefinement` (private) | `retrieval/modules/retrieval/refinement/base.py` |
| `HyDeRewriter` | `HyDeRewriting` | `retrieval/modules/retrieval/search/rewriting/hyde.py` |
| `MultiQueryRewriter` | `MultiQueryRewriting` | `.../multi_query.py` |
| `StepBackRewriter` | `StepBackRewriting` | `.../step_back.py` |
| `BaseQueryRewriter` | `BaseQueryRewriting` (private) | `.../rewriting/base.py` |
| `LLMJudge` | `LLMJudgment` | `retrieval/modules/evaluation/metrics.py` |
| `LLMRetrievalJudge` | `RetrievalJudgment` | `retrieval/modules/retrieval/judging.py` |
| `BaseRetrievalJudge` | `BaseRetrievalJudgment` (private) | `retrieval/modules/retrieval/judging.py` |
| `AceError` | `ReasoningError` | `reasoning/common/errors.py` |

**Not renamed** (left as-is):
- Config dataclasses follow `*Config` convention — already consistent.
- `Embeddings`/`Vision`/`Reranking` stay as facades (already gerund/noun form).
- Method classes (`VectorRetrieval`, `DocumentIngestion`, etc.) are already noun-form, fine.

---

## 2. Env var cleanup

### 2.1 Three layers — currently mashed together, now separated

**Layer 1: SDK env vars** (read by `x64rag` as a library)
```bash
X64RAG_LOG_ENABLED=false
X64RAG_LOG_LEVEL=INFO
X64RAG_BAML_LOG=warn       # NEW — we set os.environ["BAML_LOG"] from this internally
```
**No API keys on the SDK side.** Consumers pass them via `LanguageModelProvider(api_key=...)`.

**Layer 2: CLI env vars** (read only by `x64rag retrieval ...` / `x64rag reasoning ...`)
```bash
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
COHERE_API_KEY=
VOYAGE_API_KEY=
```
CLI loads these from `.env` to populate `LanguageModelProvider.api_key` when building services. SDK library code never reads them.

**Layer 3: BAML runtime** (internal)
- `BAML_LOG` is set internally by the SDK from `X64RAG_BAML_LOG` during startup.
- `env.OPENAI_API_KEY` default in `baml_src/clients.baml` is **removed** (keys arrive via `ClientRegistry` from `build_registry`).

### 2.2 Removed env vars

- `X64RAG_PROVIDER` — drop. Was reasoning-CLI-only. Users edit `config.toml`.
- `X64RAG_MODEL` — drop. Same.
- `X64RAG_API_KEY` — drop. Same. Redundant with vendor keys.

### 2.3 Test impact

Delete env-override tests in `reasoning/tests/test_cli_config.py` that exercise removed vars:
- `test_env_var_overrides_provider`
- `test_env_var_overrides_api_key`

### 2.4 README layout

`README.md`, `retrieval/README.md`, `reasoning/README.md` get separate "SDK Env Variables" and "CLI Env Variables" sections with clear labels.

---

## 3. Protocol deduplication

### 3.1 `BaseEmbeddings` — one definition

Currently defined twice with identical contract:
- `retrieval/modules/ingestion/embeddings/base.py:4`
- `reasoning/protocols.py:6`

**Action:** Single definition in `x64rag/common/protocols.py`. Both SDKs re-export from there via their own `common/` shims.

Since `Base*` protocols are going private per §4 (strict API trim), the `common/protocols.py` definition is only used internally by the `Embeddings` facade and by the reasoning services that need an embeddings contract.

### 3.2 `BaseVectorStore` — separate contracts, separate names

Currently defined twice with **different contracts**:
- `retrieval/stores/vector/base.py:6` — full CRUD (upsert, hybrid_search, delete, count, initialize, shutdown, retrieve, scroll, search) — stays `BaseVectorStore`.
- `reasoning/protocols.py:15` — read-only (scroll, search) — **renamed to `BaseSemanticIndex`**.

Both go private per §4. They stay in their respective SDK trees (retrieval in `stores/vector/base.py`, reasoning in `protocols.py` or similar).

---

## 4. Strict public API surface

End-user audience only — users who *configure* and *call* the SDK. Plugin-author protocols deferred.

### 4.1 Retrieval `__all__` (public)

**Entry + configs**
- `RagServer`, `RagServerConfig`
- `PersistenceConfig`, `IngestionConfig`, `RetrievalConfig`, `GenerationConfig`
- `TreeIndexingConfig`, `TreeSearchConfig`
- `ConfidenceConfig`, `BatchConfig` (if user-tunable — see §4.3)

**Providers** (from `x64rag.common`)
- `LanguageModelProvider`, `LanguageModelClient`

**Active clients**
- `Embeddings`, `Vision`, `Reranking`
- `FastEmbedSparseEmbeddings` (only concrete sparse impl; user constructs directly)

**Stores** (user-constructed)
- `QdrantVectorStore`, `Neo4jGraphStore`, `SQLAlchemyMetadataStore`
- `PostgresDocumentStore`, `FilesystemDocumentStore`

**Pluggable method classes** (user can pass explicit instances)
- `VectorIngestion`, `DocumentIngestion`, `GraphIngestion`, `TreeIngestion`
- `VectorRetrieval`, `DocumentRetrieval`, `GraphRetrieval`

**Pipeline components** (user-selectable)
- `HyDeRewriting`, `MultiQueryRewriting`, `StepBackRewriting`
- `ExtractiveRefinement`, `AbstractiveRefinement`
- `RetrievalJudgment`

**Result models** (consumed from responses)
- `QueryResult`, `StepResult`, `StreamEvent`
- `RetrievedChunk`, `ContentMatch`, `Source`, `SparseVector`
- `GraphEntity`, `GraphRelation`, `GraphPath`, `GraphResult`
- `TreeIndex`, `TreeNode`, `TreePage`, `TreeSearchResult`
- `JudgmentResult`, `MetricResult`

**Evaluation metrics** (user-run)
- `ExactMatch`, `F1Score`, `LLMJudgment`
- `RetrievalPrecision`, `RetrievalRecall`

**Errors**
- `X64RagError`, `ConfigurationError`
- `RagError`, `IngestionError`, `ParseError`, `EmptyDocumentError`, `EmbeddingError`
- `IngestionInterruptedError`, `RetrievalError`, `GenerationError`
- `StoreError`, `DuplicateSourceError`, `SourceNotFoundError`
- `TreeIndexingError`, `TreeSearchError`

### 4.2 Reasoning `__all__` (public)

**Services + pipeline**
- `AnalysisService`, `ClassificationService`, `ClusteringService`
- `ComplianceService`, `EvaluationService`
- `Pipeline`

**Providers** (from `x64rag.common`)
- `LanguageModelProvider`, `LanguageModelClient`

**Configs**
- `AnalysisConfig`, `ClassificationConfig`, `ClusteringConfig`
- `ComplianceConfig`, `EvaluationConfig`
- `ContextTrackingConfig`

**Definitions** (user-created)
- `DimensionDefinition`, `EntityTypeDefinition`
- `CategoryDefinition`, `ClassificationSetDefinition`
- `ComplianceDimensionDefinition`, `EvaluationDimensionDefinition`

**Result models**
- `AnalysisResult`, `DimensionResult`, `Entity`, `IntentShift`, `Message`, `RetrievalHint`
- `Classification`, `ClassificationSetResult`
- `Cluster`, `ClusteringResult`, `ClusterChange`, `ClusterComparison`, `TextWithMetadata`
- `ComplianceResult`, `Violation`
- `EvaluationPair`, `EvaluationReport`, `EvaluationResult`

**Pipeline steps**
- `AnalyzeStep`, `ClassifyStep`, `ComplianceStep`, `EvaluateStep`
- `PipelineResult`, `PipelineServices`

**Utilities**
- `compare_clusters`

**Errors**
- `X64RagError`, `ConfigurationError`
- `ReasoningError` (was `AceError`)
- `AnalysisError`, `ClassificationError`, `ClusteringError`
- `ComplianceError`, `EvaluationError`

### 4.3 Hidden from public API

**All `Base*` protocols:**
- `BaseEmbeddings`, `BaseVectorStore`, `BaseSemanticIndex`
- `BaseDocumentStore`, `BaseGraphStore`, `BaseMetadataStore`
- `BaseIngestionMethod`, `BaseRetrievalMethod`
- `BaseQueryRewriting`, `BaseChunkRefinement`, `BaseRetrievalJudgment`
- `BaseReranking`, `BaseVision`, `BaseSparseEmbeddings`, `BaseParser`
- `BaseMetric`, `BaseRetrievalMetric`

**Internal machinery:**
- `MethodNamespace`

**Vision-analysis intermediate models** (pipeline internals, not user-facing):
- `PageAnalysis`, `DocumentSynthesis`, `DiscoveredEntity`

**Concrete per-provider implementations** (now private behind facades):
- `_OpenAIEmbeddings`, `_VoyageEmbeddings`, `_CohereEmbeddings`
- `_AnthropicVision`, `_OpenAIVision`
- `_CohereReranking`, `_VoyageReranking`, `_LLMReranking`

**Note:** `ConfidenceConfig`, `BatchConfig` — review whether users tune these. If yes, public; if internal defaults suffice, hide. Decision deferred to implementation; default to hiding until a use case emerges.

---

## 5. Docstring / marketing corrections

### 5.1 Package docstrings

- `x64rag/__init__.py:1` — "Retrieval-Augmented Generation + Reasoning-Augmented Classification SDK" → "Retrieval-Augmented Generation + Reasoning services SDK"
- `reasoning/__init__.py:1` — "RAC — Reasoning-Augmented Classification SDK" → "Reasoning services — analysis, classification, clustering, compliance, evaluation, pipelines"
- `retrieval/__init__.py:1` — stays ("RAG — Retrieval-Augmented Generation SDK")

---

## 6. BAML clients.baml cleanup

Both `src/rfnry_rag/retrieval/baml/baml_src/clients.baml:41` and `src/rfnry_rag/reasoning/baml/baml_src/clients.baml:41` currently contain:

```baml
api_key env.OPENAI_API_KEY
```

This causes BAML to silently read vendor env vars behind the user's back. **Remove this line.** Keys arrive exclusively via `ClientRegistry` from `build_registry(lm_client)`. Regenerate BAML clients via `poe baml:generate:retrieval` and `poe baml:generate:reasoning`.

---

## 7. Examples and READMEs

**Update all 9 example files:**
- `examples/retrieval/sdk/{minimal,basic,hybrid_search,modular_pipeline}.py`
- `examples/reasoning/sdk/{minimal,basic,chat_support,email_system,quality_assurance}.py`

Replace `OpenAIEmbeddings(...)` / `LanguageModelConfig(client=LanguageModelClientConfig(...))` / etc. with the new facade + provider pattern.

**Update all READMEs:**
- `README.md` (top-level)
- `src/rfnry_rag/retrieval/README.md`
- `src/rfnry_rag/reasoning/README.md`
- `.env.example`
- `CLAUDE.md` (env var section, key patterns section)

---

## 8. Implementation ordering

Order matters — the shared primitive must exist before anything can consume it.

1. **Common primitive** — Rename `LanguageModelClientConfig` → `LanguageModelProvider`, `LanguageModelConfig` → `LanguageModelClient`. Move `max_tokens`/`temperature` from primitive to client. Update `build_registry`.
2. **Protocol dedup** — Move `BaseEmbeddings` to `common/protocols.py`. Rename reasoning's `BaseVectorStore` → `BaseSemanticIndex`.
3. **AceError rename** — `AceError` → `ReasoningError`, propagate.
4. **Embeddings facade** — Make concrete classes private, add `Embeddings` facade.
5. **Vision facade** — Make concrete classes private, add `Vision` facade.
6. **Reranking unified facade** — Make concrete classes private, add `Reranking` facade accepting union type.
7. **Refinement/Rewriting/Judgment gerundification** — rename agent nouns.
8. **BAML clients.baml** — remove `env.OPENAI_API_KEY` default lines. Regenerate.
9. **Env var cleanup** — delete `X64RAG_PROVIDER`/`X64RAG_MODEL`/`X64RAG_API_KEY` usage in `reasoning/cli/config.py`. Add `X64RAG_BAML_LOG` handling in `common/logging.py` (or startup).
10. **Public API trim** — Rewrite `__init__.py` files for strict surface.
11. **Examples + READMEs** — Update all 9 example files + 3 READMEs + `CLAUDE.md`.
12. **Tests** — Delete env-override tests. Update tests referencing renamed classes.
13. **Lint/typecheck/test** — `poe format && poe check && poe typecheck && poe test`. Fix fallout.

---

## 9. Risk / breakage

- **Pre-1.0**, no backward-compat shims. Every rename is a hard break.
- Consumers updating from the previous version will need to:
  - Rename `LanguageModelClientConfig` → `LanguageModelProvider`, `LanguageModelConfig` → `LanguageModelClient`.
  - Replace `OpenAIEmbeddings(api_key=..., model=...)` with `Embeddings(LanguageModelProvider(provider="openai", model=..., api_key=...))`.
  - Same pattern for Vision, Reranking.
  - Rename refinement/rewriting/judgment imports.
  - Rename `AceError` → `ReasoningError`.
  - Remove any reliance on `X64RAG_PROVIDER`/`X64RAG_MODEL`/`X64RAG_API_KEY`.
  - Replace `BAML_LOG` env var usage with `X64RAG_BAML_LOG`.

A migration note goes in the changelog + README release notes.

---

## Open questions (defer to implementation)

- `ConfidenceConfig`, `BatchConfig` — public or private? Default private; revisit if users ask.
- Should `x64rag.common` expose `LanguageModelProvider`/`LanguageModelClient` as the canonical import, with retrieval/reasoning shims re-exporting? Or should each SDK re-export from its own `common/` only? Default: SDK shims re-export, and top-level `x64rag/__init__.py` re-exports from `x64rag.common` directly so `from x64rag import LanguageModelProvider` works.
