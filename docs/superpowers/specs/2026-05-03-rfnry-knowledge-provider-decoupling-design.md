# rfnry-knowledge: provider decoupling + cleaning refactor

**Date:** 2026-05-03
**Status:** Approved (user-confirmed via brainstorming Q1–Q8, Approach A)

## Goal

Finish the in-flight `rfnry_rag → rfnry_knowledge` rename and strip every vendor-shaped artifact from the library. The library defines contracts (Protocols + small dataclasses); the consumer brings provider implementations and plugs them in. No `anthropic` / `openai` / `google-genai` / `voyageai` / `cohere` / `fastembed` / `tiktoken` dependencies remain. BAML stays as the structured-output spine but is fed a vendor-agnostic `ProviderClient` the consumer constructs.

## Non-goals

- Shipping reference provider implementations inside this repo (those go in a future `rfnry-providers` repo).
- Preserving the CLI.
- Maintaining backward compatibility with the `rfnry_rag` import path.
- Coupling to `rfnry-protocols` or `rfnry-providers` from this lib.

## Decisions (from Q&A)

| # | Decision |
|---|----------|
| Q1 | Keep BAML. Consumer supplies a generic `ProviderClient`; lib feeds it to `ClientRegistry`. |
| Q2 | `ProviderClient` dataclass: `name`, `model`, `api_key: SecretStr`, `options: dict`, plus retry/temperature/max_tokens/timeout. No vendor knowledge. |
| Q3 | Vision routes through BAML (drawing/page extraction is structured anyway). Embeddings + Reranking become Protocols the consumer implements. |
| Q4 | `FastEmbedSparseEmbeddings` deleted. Sparse embeddings is a Protocol; consumer brings their own. |
| Q5 | `tiktoken` deleted. `TokenCounter` is a Protocol; consumer plugs one in. AUTO routing + token-mode chunking fail with `ConfigurationError` if not supplied. |
| Q6 | Vendor usage extractors deleted. BAML LLM calls read usage from BAML's response/event metadata, normalized into `TokenUsage`. |
| Q7 | CLI deleted entirely. `src/rfnry_knowledge/cli/`, `[project.scripts]`, `[project.optional-dependencies].cli`, `click` dep all removed. |
| Q8 | `lm_client` field renamed to `provider_client: ProviderClient` in `KnowledgeEngineConfig`, `DocumentExpansionConfig`, `ContextualChunkConfig`. |

## Public surface (after)

### `rfnry_knowledge.providers`

```python
@dataclass(frozen=True)
class ProviderClient:
    name: str                       # BAML provider key (consumer chooses; "anthropic" / "openai" / "my-cluster")
    model: str
    api_key: SecretStr
    options: dict[str, Any] = field(default_factory=dict)
    max_retries: int = 3
    max_tokens: int = 4096
    temperature: float = 0.0
    timeout_seconds: int = 60
    fallback: ProviderClient | None = None
    strategy: Literal["primary_only", "fallback"] = "primary_only"

class TokenUsage(TypedDict, total=False):
    input: int
    output: int
    cache_creation: int
    cache_read: int

@dataclass(frozen=True)
class EmbeddingResult:
    vectors: list[list[float]]
    usage: TokenUsage | None = None

@dataclass(frozen=True)
class RerankResult:
    chunks: list[RetrievedChunk]
    usage: TokenUsage | None = None

class BaseEmbeddings(Protocol):
    @property
    def name(self) -> str: ...
    @property
    def model(self) -> str: ...
    async def embed(self, texts: list[str]) -> EmbeddingResult: ...
    async def embedding_dimension(self) -> int: ...

class BaseSparseEmbeddings(Protocol):
    @property
    def name(self) -> str: ...
    @property
    def model(self) -> str: ...
    async def embed_sparse(self, texts: list[str]) -> list[SparseVector]: ...

class BaseReranking(Protocol):
    @property
    def name(self) -> str: ...
    @property
    def model(self) -> str: ...
    async def rerank(self, query: str, results: list[RetrievedChunk], top_k: int) -> RerankResult: ...

class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...
```

### `rfnry_knowledge` top-level (re-exports)

Add: `ProviderClient`, `TokenUsage`, `EmbeddingResult`, `RerankResult`, `BaseEmbeddings`, `BaseSparseEmbeddings`, `BaseReranking`, `TokenCounter`.

Remove: `AnthropicModelProvider`, `OpenAIModelProvider`, `GoogleModelProvider`, `VoyageModelProvider`, `CohereModelProvider`, `LLMClient`, `Embeddings`, `Vision`, `Reranking`, `FastEmbedSparseEmbeddings`.

## Code deletions

| Path | Action |
|------|--------|
| `src/rfnry_knowledge/providers/provider.py` | rewrite — replace `_BaseModelProvider` + 5 vendor classes with `ProviderClient` |
| `src/rfnry_knowledge/providers/client.py` | delete (`LLMClient` superseded by `ProviderClient`) |
| `src/rfnry_knowledge/providers/facades.py` | delete (`Embeddings`, `Vision`, `Reranking` facades) |
| `src/rfnry_knowledge/providers/text_generation.py` | rewrite — single BAML-backed path, no vendor SDK imports |
| `src/rfnry_knowledge/providers/protocols.py` | extend — add new Protocols + dataclasses |
| `src/rfnry_knowledge/providers/registry.py` | rewrite — accept `ProviderClient` |
| `src/rfnry_knowledge/ingestion/embeddings/openai.py` | delete |
| `src/rfnry_knowledge/ingestion/embeddings/cohere.py` | delete |
| `src/rfnry_knowledge/ingestion/embeddings/voyage.py` | delete |
| `src/rfnry_knowledge/ingestion/embeddings/sparse/fastembed.py` | delete |
| `src/rfnry_knowledge/ingestion/vision/anthropic.py` | delete |
| `src/rfnry_knowledge/ingestion/vision/openai.py` | delete |
| `src/rfnry_knowledge/ingestion/vision/gemini.py` | delete |
| `src/rfnry_knowledge/retrieval/search/reranking/cohere.py` | delete |
| `src/rfnry_knowledge/retrieval/search/reranking/voyage.py` | delete |
| `src/rfnry_knowledge/ingestion/chunk/token_counter.py` | rewrite — Protocol-only, tiktoken impl removed |
| `src/rfnry_knowledge/cli/` | delete entire tree |
| `src/rfnry_knowledge/telemetry/usage.py` | rewrite — drop `extract_anthropic_usage`/`extract_openai_usage`/`extract_gemini_usage`, keep `add_llm_usage` only |
| `tests/retrieval/test_vision_*.py` | delete vendor-impl tests; keep BAML-routed Vision tests |
| `tests/` | retarget all `from rfnry_rag` imports |

## Telemetry contract

Single normalized vocabulary across BAML and consumer-supplied Protocols:

```python
TokenUsage = TypedDict("TokenUsage", {"input": int, "output": int, "cache_creation": int, "cache_read": int}, total=False)
```

Lib accumulates into the active `QueryTelemetryRow` / `IngestTelemetryRow` via `add_llm_usage(provider, model, usage)` (existing). Consumer-supplied Protocols return `EmbeddingResult` / `RerankResult` with optional `usage`; `None` or missing keys → zeros, never raise.

## pyproject.toml

**Drop:** `openai`, `anthropic`, `google-genai`, `voyageai`, `cohere`, `fastembed`, `tiktoken`.
**Drop:** `[project.optional-dependencies].cli` group, `click` transitive.
**Drop:** `[project.scripts]` entry.
**Keep:** `baml-py`, `qdrant-client`, `sqlalchemy[asyncio]`, `asyncpg`, `aiosqlite`, `pymupdf`, `rank-bm25`, `pydantic`, `scikit-learn`, `lxml`, `ezdxf`, `matplotlib`.
**Keep:** `[project.optional-dependencies].graph` (`neo4j`).
**Update:** ruff per-file-ignores — drop CLI lines.

## Validation

- `poe typecheck` clean (no vendor-SDK imports remain).
- `poe test` passes against the trimmed test suite.
- `grep -r "from anthropic\|from openai\|from google\|import voyageai\|from cohere\|import fastembed\|import tiktoken" src/` returns nothing.
- `grep -r "rfnry_rag" src/ tests/` returns nothing.
- `grep -r "AnthropicModelProvider\|OpenAIModelProvider\|GoogleModelProvider\|VoyageModelProvider\|CohereModelProvider\|LLMClient\b\|FastEmbedSparseEmbeddings" src/` returns nothing.
- `grep -r "rfnry_protocols\|rfnry_providers\|rfnry-protocols\|rfnry-providers" .` returns nothing.

## Execution

Approach A — single branch (current `main`), one cleaning sweep, commit + push when validation passes. Implementation plan written via `superpowers:writing-plans` skill, executed task-by-task.
