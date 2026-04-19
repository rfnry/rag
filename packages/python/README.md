## rfnry-rag — retrieval augmented generation engine

Lexical/Semantical Retrieval Engine

- **retrieval** — lexical and semantical retrieval engine.
- **reasoning** — analysis, classification, clustering, compliance, evaluation, toolkit.

[retrieval](src/rfnry-rag/retrieval/README.md) · [reasoning](src/rfnry-rag/reasoning/README.md) · [examples](examples)

## Fundamentals

- **Modular** Retrieval and ingestion methods are pluggable via `BaseRetrievalMethod` / `BaseIngestionMethod` protocols. Vector, document, graph and tree paths are all optional; at least one must be configured. No mandatory vector DB, no mandatory embeddings. Reasoning services (`AnalysisService`, `ClassificationService`, `ClusteringService`, `ComplianceService`, `EvaluationService`) are standalone. Use one, compose several through `Pipeline`, or wire them into a retrieval flow.
- **Protocol** `BaseEmbeddings`, `BaseSemanticIndex`, `BaseReranking`, `BaseQueryRewriting`, `BaseChunkRefinement` — any conforming object works. Swap components without touching the engine.
- **Multi-path** Configured methods run in parallel per query. Results merge via reciprocal rank fusion with per-method weights. Per-method error isolation: one path fails, the rest continue.
- **Pipeline** Query rewriting → multi-path search → reranking → chunk refinement → grounding and generation. Each stage is optional and independently configurable.
- **BAML** Structured output parsing, retry and fallback policies, primary + fallback provider routing via `LanguageModelClient`, observability through Boundary Studio or `baml_py.Collector`.
- **Unified CLI** `rfnry-rag retrieval …` and `rfnry-rag reasoning …` mirror the SDK surface for scripting and inspection.

## Installation

```bash
uv add rfnry-rag                    # core SDK
uv add "rfnry_rag[graph]"           # + Neo4j graph support
uv add "rfnry-rag[cli]"             # + CLI
```

## Retrieval getting started

```python
from rfnry_rag.retrieval import RagEngine, RagServerConfig, PersistenceConfig, IngestionConfig
from rfnry_rag.retrieval import QdrantVectorStore, Embeddings
from rfnry_rag import LanguageModelProvider

config = RagServerConfig(
    persistence=PersistenceConfig(
        vector_store=QdrantVectorStore(url="http://localhost:6333", collection="docs"),
    ),
    ingestion=IngestionConfig(
        embeddings=Embeddings(LanguageModelProvider(
            provider="openai", model="text-embedding-3-small", api_key="...",
        )),
    ),
)

async with RagEngine(config) as rag:
    await rag.ingest("manual.pdf", knowledge_id="equipment")
    await rag.ingest("annual_report.pdf", knowledge_id="reports", tree_index=True)
    result = await rag.query("How do I replace the filter?", knowledge_id="equipment")
    print(result.answer)
```

Fine-grained method access:

```python
async with RagEngine(config) as rag:
    vector_chunks = await rag.retrieval.vector.search("pressure specs", top_k=20)
    doc_chunks    = await rag.retrieval.document.search("pressure specs", top_k=10)
    result        = await rag.query("What are the pressure specifications?", knowledge_id="equipment")
```

CLI:

```bash
rfnry-rag retrieval init
rfnry-rag retrieval ingest manual.pdf -k equipment
rfnry-rag retrieval query "how to replace the filter?" -k equipment
rfnry-rag retrieval retrieve "part number 8842-A" -k equipment
```

## Reasoning getting started

Each service is standalone. Compose them through `Pipeline` when you need sequential steps.

```python
from rfnry_rag.reasoning import AnalysisService, AnalysisConfig, DimensionDefinition
from rfnry_rag import LanguageModelClient, LanguageModelProvider

lm = LanguageModelClient(
    provider=LanguageModelProvider(
        provider="anthropic", model="claude-sonnet-4-20250514", api_key="...",
    ),
)

analyzer = AnalysisService(lm_client=lm)
result = await analyzer.analyze(
    "My order FB-12345 hasn't arrived and I need it by Friday.",
    config=AnalysisConfig(
        dimensions=[DimensionDefinition("urgency", "How time-sensitive", "0.0-1.0")],
        summarize=True,
    ),
)
print(f"{result.primary_intent} — urgency: {result.dimensions['urgency'].value}")
```

CLI:

```bash
rfnry-rag reasoning init
rfnry-rag reasoning analyze "My order FB-12345 hasn't arrived and I need it by Friday"
rfnry-rag reasoning classify "I want my money back" --categories categories.json
rfnry-rag reasoning compliance "We'll give you 150% refund" --references policy.md
```

Runnable examples: [`examples/retrieval/sdk`](examples/retrieval/sdk), [`examples/reasoning/sdk`](examples/reasoning/sdk), and their `cli/` counterparts.

## Development Setup

All tasks run via [poethepoet](https://github.com/nat-n/poethepoet):

```bash
uv sync --extra dev                   # dev extras only
uv sync --all-extras                  # everything

uv run poe format                     # ruff format
uv run poe check                      # ruff lint
uv run poe check:fix                  # ruff lint + auto-fix
uv run poe typecheck                  # mypy
uv run poe test                       # pytest
uv run poe baml:generate:retrieval    # regenerate retrieval BAML client
uv run poe baml:generate:reasoning    # regenerate reasoning BAML client
```

## Observability

All LLM calls go through [BAML](https://docs.boundaryml.com/).

- **Boundary Studio** — set `boundary_api_key` on a `LanguageModelClient` to enable automatic cloud tracing with token counts, latency, and function-level tracking.
- **Programmatic** — use `baml_py.Collector` for in-process token usage tracking.

## Environment variables

SDK (read when used as a library):

```bash
RRAG_LOG_ENABLED=false    # true / false
RRAG_LOG_LEVEL=INFO       # DEBUG, INFO, WARNING, ERROR
RRAG_BAML_LOG=warn        # info, warn, debug — BAML runtime log level
```

CLI only (never read by the SDK — pass API keys explicitly via `LanguageModelProvider(api_key=...)`):

```bash
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
COHERE_API_KEY=
VOYAGE_API_KEY=
```

## License

MIT — see [`LICENSE`](./LICENSE).
