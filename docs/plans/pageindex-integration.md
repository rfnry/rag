# Tree Retrieval Module — Design Document

Tree-based retrieval for x64rag, inspired by PageIndex. Adds LLM reasoning over document structure as an optional retrieval path alongside vector/BM25/graph search.

## Problem Statement

Vector search finds content that *looks similar* to a query. It works well for straightforward questions where the answer lives in a single chunk. It fails when:

- The answer spans multiple sections that need to be understood together
- The query requires understanding *where* in a document something is discussed
- The document is long and structured (reports, contracts, filings) and artificial chunking destroys the navigable hierarchy
- Similarity and relevance diverge — a methodology section might use the same words as a findings section, but only one answers the question

Tree search finds content that *is relevant* to a query. An LLM reads the document's natural structure (sections, subsections, page ranges, summaries) and reasons about where the answer lives — the same way a human would scan a table of contents before flipping to the right chapter.

## When to Use Which

| Scenario | Vector | Tree | Both |
|----------|--------|------|------|
| Short documents, simple questions | Best | Overkill | - |
| Long structured docs (reports, filings, contracts) | Weak | Best | Ideal |
| High accuracy matters, cost secondary | - | Best | Ideal |
| Low latency, high throughput | Best | Slow | - |
| Explainable retrieval needed | Weak | Best | - |

When combined, vector search provides fast broad recall, tree search provides precise reasoning-based retrieval, RRF merges both, and the generation layer gets the best of both worlds.

## Design Decisions

1. **Standalone module in retrieval, not a new search flag.** Tree search requires a completely different ingestion process (tree building vs chunking + embedding) and a different retrieval paradigm (LLM reasoning traversal vs similarity search). It earns its own module, not just a config flag.

2. **Shared generation layer.** Independent ingestion and retrieval, but once tree retrieval finds relevant pages, results normalize into `SearchResult` format and feed into the existing generation pipeline (grounding gate, relevance gate, generation). The last mile is shared.

3. **Tree stored in metadata store (JSONB).** The tree is a single JSON document per source — a JSONB column in the existing metadata store. No dedicated tree store needed. Loaded into memory for retrieval (10-20KB per document).

4. **Opt-in at ingestion time.** `tree_index=True` when ingesting. Both pipelines (chunk+embed AND tree building) run on the same document. Tree indexing is expensive (multiple LLM calls), so users pay the cost consciously per document.

5. **Configurable via RetrievalConfig.** `tree_search=True` activates tree retrieval as another search path. Results merge with vector/BM25/graph via RRF. Users control cost per query.

6. **Two model configs.** One for indexing (high-volume, simpler tasks), one for retrieval reasoning (low-volume, harder task). Follows the existing pattern where each step has its own `LanguageModelConfig` slot.

7. **BAML tool-use loop for retrieval.** The LLM chooses actions (`ToolFetchPages`, `ToolDrillDown`, `ToolResolvedPages`) via typed union returns. Python orchestrates the loop. Simple queries resolve in 1 step, complex documents get multi-step traversal. Cost bounded by `max_steps`.

## Module Structure

```
retrieval/modules/
  ingestion/
    tree/                          # NEW — tree index building
      __init__.py
      service.py                   # TreeIndexingService
      toc.py                       # TOC detection, parsing, verification
      structure.py                 # tree building, node splitting, post-processing
  retrieval/
    tree/                          # NEW — tree-based search
      __init__.py
      service.py                   # TreeSearchService (BAML tool-use loop)
      tools.py                     # tool execution handlers
```

Shared models (`TreeNode`, `TreeIndex`, `TreeSearchResult`) live in `retrieval/common/models.py`.

## Data Model

```python
@dataclass
class TreeNode:
    node_id: str                    # unique ID, e.g. "0001"
    title: str                      # section title
    start_index: int                # first page (PDF) or line (MD)
    end_index: int                  # last page/line
    summary: str | None = None      # LLM-generated summary
    children: list[TreeNode] = field(default_factory=list)

@dataclass
class TreeIndex:
    source_id: str                  # links to existing document/source
    doc_name: str
    doc_description: str | None     # one-line LLM summary
    structure: list[TreeNode]       # root-level nodes
    page_count: int
    created_at: datetime

@dataclass
class TreeSearchResult:
    node_id: str
    title: str
    pages: str                      # "5-7,12"
    content: str                    # extracted page text
    reasoning: str                  # LLM's reasoning for relevance
```

Key decisions:
- `TreeNode` is recursive — children are `list[TreeNode]`, matching natural hierarchy
- `TreeIndex` links to `source_id` — same identity as existing ingestion
- `TreeSearchResult` carries `reasoning` — the explainability advantage over vector search
- No `text` on `TreeNode` — full page text fetched on demand during search, keeps tree lightweight

## Indexing Pipeline

```
PDF/Markdown input
  |
1. Page extraction — parse pages, count tokens per page
  |
2. TOC detection — scan first N pages for table of contents
  |-> TOC with page numbers -> parse structure, map to physical pages, apply offset
  |-> TOC without page numbers -> parse structure, LLM finds each section start
  |-> No TOC -> LLM extracts structure from page content in groups
  |
3. Position verification — concurrent LLM calls confirming sections appear where expected
  |
4. Tree building — convert flat TOC list to recursive TreeNode hierarchy, calculate page ranges
  |
5. Large node splitting — recursively subdivide nodes exceeding max_pages / max_tokens
  |
6. Summary generation (optional) — concurrent LLM summaries per node
  |
7. Doc description (optional) — single LLM call for one-line document summary
  |
TreeIndex stored in metadata store (JSONB)
```

### Indexing Configuration

```python
@dataclass
class TreeIndexingConfig:
    enabled: bool = False
    model: LanguageModelConfig | None = None     # defaults to SDK's main model
    toc_scan_pages: int = 20
    max_pages_per_node: int = 10
    max_tokens_per_node: int = 20_000
    generate_summaries: bool = True
    generate_description: bool = True
```

## Search Pipeline

```
User query arrives with tree_search enabled in RetrievalConfig
  |
1. Load TreeIndex from metadata store for relevant sources
  |
2. Prepare tree for prompt — serialize to compact string
   (node IDs, titles, page ranges, summaries — no full text)
  |
3. BAML tool-use loop (max_steps configurable):
  +---------------------------------------------+
  | Call TreeRetrievalStep(query, tree, context) |
  |   |                                         |
  | ToolFetchPages -> fetch page text, append   |
  |                   to context, loop again    |
  |   |                                         |
  | ToolDrillDown -> replace tree with subtree, |
  |                  loop again                 |
  |   |                                         |
  | ToolResolvedPages -> exit loop              |
  +---------------------------------------------+
  |
4. Build TreeSearchResult(s) from resolved pages
  |
5. Normalize into SearchResult format (same as vector/BM25 results)
  |
6. Return to multi-path merger -> RRF with other search paths -> generation
```

### Search Configuration

```python
@dataclass
class TreeSearchConfig:
    enabled: bool = False
    model: LanguageModelConfig | None = None     # defaults to SDK's main model
    max_steps: int = 5
    max_context_tokens: int = 50_000
```

## BAML Definitions

### Tool Types

```baml
class ToolFetchPages {
    pages string @description("page ranges to retrieve, e.g. '5-7' or '3,8,12-15'")
    reasoning string @description("why these pages are likely relevant to the query")
}

class ToolDrillDown {
    node_id string @description("ID of the node to explore deeper")
    reasoning string @description("why this subtree needs closer inspection")
}

class ToolResolvedPages {
    pages string @description("final relevant page ranges that answer the query")
    reasoning string @description("summary of how these pages address the query")
}
```

### Retrieval Function

```baml
function TreeRetrievalStep(
    query: string,
    tree_structure: string,
    accumulated_context: string
) -> ToolFetchPages | ToolDrillDown | ToolResolvedPages {
    client TreeRetrievalModel
    prompt #"
        You are a document retrieval specialist. Given a query and a
        document's hierarchical structure, decide your next action.

        QUERY: {{ query }}

        DOCUMENT STRUCTURE:
        {{ tree_structure }}

        {% if accumulated_context %}
        PAGES ALREADY RETRIEVED:
        {{ accumulated_context }}
        {% endif %}

        ACTIONS:
        - ToolFetchPages: retrieve specific pages to gather evidence
        - ToolDrillDown: zoom into a subtree node for deeper inspection
        - ToolResolvedPages: you have enough information, declare final pages

        Think step by step about which sections are relevant, then choose
        your action.

        {{ ctx.output_format }}
    "#
}
```

### Indexing Functions

```baml
function DetectTableOfContents(page_text: string) -> TocDetectionResult {
    client TreeIndexingModel
    prompt #"..."#
}

function ExtractStructure(pages_text: string) -> TocStructure {
    client TreeIndexingModel
    prompt #"..."#
}

function VerifySectionPosition(title: string, page_text: string) -> bool {
    client TreeIndexingModel
    prompt #"..."#
}

function GenerateNodeSummary(section_text: string) -> string {
    client TreeIndexingModel
    prompt #"..."#
}

function GenerateDocDescription(tree_structure: string) -> string {
    client TreeIndexingModel
    prompt #"..."#
}
```

## Integration with RagServer

### Configuration

```python
class RagConfig:
    # ... existing fields ...
    tree_indexing: TreeIndexingConfig | None = None
    tree_search: TreeSearchConfig | None = None
```

### Ingestion

```python
async def ingest(self, source, *, tree_index: bool = False, **kwargs):
    # ... existing chunking, embedding, analysis ...

    if tree_index and self.config.tree_indexing:
        tree = await self.tree_indexing_service.index(source)
        await self.metadata_store.save_tree_index(source_id, tree)
```

### Search

```python
async def _search_paths(self, query, config):
    paths = []
    paths.append(self._vector_search(query))
    if config.bm25_enabled:
        paths.append(self._bm25_search(query))
    if config.graph_enabled:
        paths.append(self._graph_search(query))
    if config.tree_search_enabled:
        paths.append(self._tree_search(query))

    results = await asyncio.gather(*paths)
    return reciprocal_rank_fusion(results)
```

### Generation

Unchanged. RRF merges tree results with other paths, generation receives `SearchResult` list as always.

## What We Take from PageIndex

- Full TOC detection logic (three paths: TOC with page numbers, TOC without page numbers, no TOC)
- Position verification (concurrent LLM checks)
- Recursive large-node splitting
- Summary generation for tree navigation
- Page-level content retrieval

## What We Adapt

- LLM calls go through BAML instead of raw LiteLLM
- Config follows existing x64rag patterns
- Storage uses metadata store (JSONB) instead of filesystem JSON
- Error handling uses existing `IngestionError` / `RetrievalError` hierarchy
- Results normalize into `SearchResult` for RRF fusion with other paths
- Concurrency uses existing `run_concurrent` helper
