# Tree Retrieval Module — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add tree-based retrieval as an optional search path in the retrieval SDK, using LLM reasoning over document structure instead of vector similarity.

**Architecture:** Tree indexing builds a hierarchical section tree from documents (via TOC detection or LLM extraction) and stores it as JSONB in the metadata store. Tree search uses a BAML tool-use loop where the LLM traverses the tree to find relevant pages. Results normalize into `RetrievedChunk` and merge with vector/BM25/graph results via RRF.

**Tech Stack:** BAML (structured LLM calls), PyPDF2/PyMuPDF (page extraction), SQLAlchemy (JSONB storage), asyncio (concurrency)

**Design doc:** `docs/plans/pageindex-integration.md`

---

### Task 1: Data Models

Add tree-specific dataclasses to the existing models file.

**Files:**
- Modify: `src/rfnry_rag/retrieval/common/models.py`
- Test: `src/rfnry_rag/retrieval/tests/test_tree_models.py`

**Step 1: Write the failing test**

```python
"""Tests for tree data models."""

from datetime import datetime, timezone

from x64rag.retrieval.common.models import TreeIndex, TreeNode, TreeSearchResult


def test_tree_node_defaults():
    node = TreeNode(node_id="0001", title="Chapter 1", start_index=1, end_index=10)
    assert node.summary is None
    assert node.children == []


def test_tree_node_with_children():
    child = TreeNode(node_id="0002", title="Section 1.1", start_index=1, end_index=5)
    parent = TreeNode(
        node_id="0001", title="Chapter 1", start_index=1, end_index=10, children=[child]
    )
    assert len(parent.children) == 1
    assert parent.children[0].node_id == "0002"


def test_tree_node_to_dict():
    child = TreeNode(node_id="0002", title="Section 1.1", start_index=1, end_index=5, summary="Sub")
    node = TreeNode(
        node_id="0001",
        title="Chapter 1",
        start_index=1,
        end_index=10,
        summary="Top",
        children=[child],
    )
    d = node.to_dict()
    assert d["node_id"] == "0001"
    assert d["children"][0]["node_id"] == "0002"


def test_tree_node_from_dict():
    data = {
        "node_id": "0001",
        "title": "Chapter 1",
        "start_index": 1,
        "end_index": 10,
        "summary": "Top",
        "children": [
            {
                "node_id": "0002",
                "title": "Section 1.1",
                "start_index": 1,
                "end_index": 5,
                "summary": "Sub",
                "children": [],
            }
        ],
    }
    node = TreeNode.from_dict(data)
    assert node.node_id == "0001"
    assert node.children[0].title == "Section 1.1"


def test_tree_index_to_dict():
    now = datetime.now(timezone.utc)
    node = TreeNode(node_id="0001", title="Chapter 1", start_index=1, end_index=10)
    index = TreeIndex(
        source_id="src-1",
        doc_name="report.pdf",
        doc_description="Annual report",
        structure=[node],
        page_count=10,
        created_at=now,
    )
    d = index.to_dict()
    assert d["source_id"] == "src-1"
    assert d["structure"][0]["node_id"] == "0001"
    assert d["created_at"] == now.isoformat()


def test_tree_index_from_dict():
    data = {
        "source_id": "src-1",
        "doc_name": "report.pdf",
        "doc_description": "Annual report",
        "structure": [
            {"node_id": "0001", "title": "Ch1", "start_index": 1, "end_index": 10, "children": []}
        ],
        "page_count": 10,
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    index = TreeIndex.from_dict(data)
    assert index.source_id == "src-1"
    assert index.structure[0].node_id == "0001"


def test_tree_search_result():
    r = TreeSearchResult(
        node_id="0001",
        title="Chapter 1",
        pages="5-7,12",
        content="Page text here",
        reasoning="This section covers the topic",
    )
    assert r.pages == "5-7,12"
    assert r.reasoning == "This section covers the topic"
```

**Step 2: Run test to verify it fails**

Run: `pytest src/rfnry_rag/retrieval/tests/test_tree_models.py -v`
Expected: FAIL — `TreeNode`, `TreeIndex`, `TreeSearchResult` not found

**Step 3: Write the implementation**

Add to `src/rfnry_rag/retrieval/common/models.py`:

```python
@dataclass
class TreeNode:
    """A node in the document tree index."""

    node_id: str
    title: str
    start_index: int
    end_index: int
    summary: str | None = None
    children: list[TreeNode] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "title": self.title,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "summary": self.summary,
            "children": [c.to_dict() for c in self.children],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TreeNode:
        return cls(
            node_id=data["node_id"],
            title=data["title"],
            start_index=data["start_index"],
            end_index=data["end_index"],
            summary=data.get("summary"),
            children=[cls.from_dict(c) for c in data.get("children", [])],
        )


@dataclass
class TreeIndex:
    """Complete tree index for a document."""

    source_id: str
    doc_name: str
    doc_description: str | None
    structure: list[TreeNode]
    page_count: int
    created_at: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "doc_name": self.doc_name,
            "doc_description": self.doc_description,
            "structure": [n.to_dict() for n in self.structure],
            "page_count": self.page_count,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TreeIndex:
        return cls(
            source_id=data["source_id"],
            doc_name=data["doc_name"],
            doc_description=data.get("doc_description"),
            structure=[TreeNode.from_dict(n) for n in data["structure"]],
            page_count=data["page_count"],
            created_at=datetime.fromisoformat(data["created_at"]),
        )


@dataclass
class TreeSearchResult:
    """Result from tree-based search."""

    node_id: str
    title: str
    pages: str
    content: str
    reasoning: str
```

**Step 4: Run test to verify it passes**

Run: `pytest src/rfnry_rag/retrieval/tests/test_tree_models.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/rfnry_rag/retrieval/common/models.py src/rfnry_rag/retrieval/tests/test_tree_models.py
git commit -m "feat(tree): add TreeNode, TreeIndex, TreeSearchResult data models"
```

---

### Task 2: Config Dataclasses

**Files:**
- Modify: `src/rfnry_rag/retrieval/server.py` (add to existing config dataclasses)
- Test: `src/rfnry_rag/retrieval/tests/test_tree_config.py`

**Step 1: Write the failing test**

```python
"""Tests for tree config validation."""

import pytest

from x64rag.common.errors import ConfigurationError
from x64rag.retrieval.server import TreeIndexingConfig, TreeSearchConfig


def test_tree_indexing_config_defaults():
    config = TreeIndexingConfig()
    assert config.enabled is False
    assert config.model is None
    assert config.toc_scan_pages == 20
    assert config.max_pages_per_node == 10
    assert config.max_tokens_per_node == 20_000
    assert config.generate_summaries is True
    assert config.generate_description is True


def test_tree_indexing_config_validation_toc_scan_pages():
    with pytest.raises(ConfigurationError, match="toc_scan_pages"):
        TreeIndexingConfig(toc_scan_pages=0)


def test_tree_indexing_config_validation_max_pages():
    with pytest.raises(ConfigurationError, match="max_pages_per_node"):
        TreeIndexingConfig(max_pages_per_node=0)


def test_tree_indexing_config_validation_max_tokens():
    with pytest.raises(ConfigurationError, match="max_tokens_per_node"):
        TreeIndexingConfig(max_tokens_per_node=0)


def test_tree_search_config_defaults():
    config = TreeSearchConfig()
    assert config.enabled is False
    assert config.model is None
    assert config.max_steps == 5
    assert config.max_context_tokens == 50_000


def test_tree_search_config_validation_max_steps():
    with pytest.raises(ConfigurationError, match="max_steps"):
        TreeSearchConfig(max_steps=0)


def test_tree_search_config_validation_max_context_tokens():
    with pytest.raises(ConfigurationError, match="max_context_tokens"):
        TreeSearchConfig(max_context_tokens=0)
```

**Step 2: Run test to verify it fails**

Run: `pytest src/rfnry_rag/retrieval/tests/test_tree_config.py -v`
Expected: FAIL — `TreeIndexingConfig`, `TreeSearchConfig` not found

**Step 3: Write the implementation**

Add to `src/rfnry_rag/retrieval/server.py` alongside the other config dataclasses:

```python
@dataclass
class TreeIndexingConfig:
    """Configuration for tree-based document indexing."""

    enabled: bool = False
    model: LanguageModelConfig | None = None
    toc_scan_pages: int = 20
    max_pages_per_node: int = 10
    max_tokens_per_node: int = 20_000
    generate_summaries: bool = True
    generate_description: bool = True

    def __post_init__(self) -> None:
        if self.toc_scan_pages < 1:
            raise ConfigurationError("toc_scan_pages must be positive")
        if self.max_pages_per_node < 1:
            raise ConfigurationError("max_pages_per_node must be positive")
        if self.max_tokens_per_node < 1:
            raise ConfigurationError("max_tokens_per_node must be positive")


@dataclass
class TreeSearchConfig:
    """Configuration for tree-based search."""

    enabled: bool = False
    model: LanguageModelConfig | None = None
    max_steps: int = 5
    max_context_tokens: int = 50_000

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ConfigurationError("max_steps must be positive")
        if self.max_context_tokens < 1:
            raise ConfigurationError("max_context_tokens must be positive")
```

Add fields to `RagServerConfig`:

```python
@dataclass
class RagServerConfig:
    persistence: PersistenceConfig
    ingestion: IngestionConfig
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    tree_indexing: TreeIndexingConfig = field(default_factory=TreeIndexingConfig)
    tree_search: TreeSearchConfig = field(default_factory=TreeSearchConfig)
```

**Step 4: Run test to verify it passes**

Run: `pytest src/rfnry_rag/retrieval/tests/test_tree_config.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/rfnry_rag/retrieval/server.py src/rfnry_rag/retrieval/tests/test_tree_config.py
git commit -m "feat(tree): add TreeIndexingConfig and TreeSearchConfig"
```

---

### Task 3: Error Types

**Files:**
- Modify: `src/rfnry_rag/retrieval/common/errors.py`
- Test: `src/rfnry_rag/retrieval/tests/test_tree_errors.py`

**Step 1: Write the failing test**

```python
"""Tests for tree error types."""

from x64rag.retrieval.common.errors import (
    IngestionError,
    RagError,
    RetrievalError,
    TreeIndexingError,
    TreeSearchError,
)


def test_tree_indexing_error_is_ingestion_error():
    err = TreeIndexingError("failed to build tree")
    assert isinstance(err, IngestionError)
    assert isinstance(err, RagError)
    assert str(err) == "failed to build tree"


def test_tree_search_error_is_retrieval_error():
    err = TreeSearchError("search loop exceeded max steps")
    assert isinstance(err, RetrievalError)
    assert isinstance(err, RagError)
    assert str(err) == "search loop exceeded max steps"
```

**Step 2: Run test to verify it fails**

Run: `pytest src/rfnry_rag/retrieval/tests/test_tree_errors.py -v`
Expected: FAIL — `TreeIndexingError`, `TreeSearchError` not found

**Step 3: Write the implementation**

Add to `src/rfnry_rag/retrieval/common/errors.py`:

```python
class TreeIndexingError(IngestionError):
    """Error during tree index construction."""


class TreeSearchError(RetrievalError):
    """Error during tree-based search."""
```

**Step 4: Run test to verify it passes**

Run: `pytest src/rfnry_rag/retrieval/tests/test_tree_errors.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/rfnry_rag/retrieval/common/errors.py src/rfnry_rag/retrieval/tests/test_tree_errors.py
git commit -m "feat(tree): add TreeIndexingError and TreeSearchError"
```

---

### Task 4: Metadata Store — Tree Index Persistence

Add a JSONB column to store tree indexes and methods to save/load them.

**Files:**
- Modify: `src/rfnry_rag/retrieval/stores/metadata/sqlalchemy.py`
- Modify: `src/rfnry_rag/retrieval/stores/metadata/base.py`
- Test: `src/rfnry_rag/retrieval/tests/test_tree_metadata.py`

**Step 1: Write the failing test**

```python
"""Tests for tree index persistence in metadata store."""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from x64rag.retrieval.common.models import TreeIndex, TreeNode


def test_tree_index_serialization_roundtrip():
    """Verify TreeIndex survives JSON serialization for JSONB storage."""
    now = datetime.now(timezone.utc)
    child = TreeNode(node_id="0002", title="Section 1.1", start_index=1, end_index=5, summary="Sub")
    root = TreeNode(
        node_id="0001", title="Chapter 1", start_index=1, end_index=10,
        summary="Top", children=[child],
    )
    index = TreeIndex(
        source_id="src-1",
        doc_name="report.pdf",
        doc_description="Annual report",
        structure=[root],
        page_count=10,
        created_at=now,
    )

    serialized = json.dumps(index.to_dict())
    deserialized = TreeIndex.from_dict(json.loads(serialized))

    assert deserialized.source_id == "src-1"
    assert deserialized.doc_name == "report.pdf"
    assert deserialized.structure[0].children[0].title == "Section 1.1"
    assert deserialized.created_at == now
```

**Step 2: Run test to verify it passes** (this one tests the models, should pass already)

Run: `pytest src/rfnry_rag/retrieval/tests/test_tree_metadata.py -v`
Expected: PASS

**Step 3: Add tree_index_json column to _SourceRow**

In `src/rfnry_rag/retrieval/stores/metadata/sqlalchemy.py`, add to `_SourceRow`:

```python
tree_index_json: Mapped[str | None] = mapped_column(Text, nullable=True)
```

The existing `_migrate_missing_columns` method will automatically add this column to existing databases.

**Step 4: Add save/load methods to the metadata store**

Add to `BaseMetadataStore` protocol in `src/rfnry_rag/retrieval/stores/metadata/base.py`:

```python
async def save_tree_index(self, source_id: str, tree_index_json: str) -> None: ...
async def get_tree_index(self, source_id: str) -> str | None: ...
```

Add implementations in the SQLAlchemy store:

```python
async def save_tree_index(self, source_id: str, tree_index_json: str) -> None:
    await self.update_source(source_id, tree_index_json=tree_index_json)

async def get_tree_index(self, source_id: str) -> str | None:
    source_row = await self._get_source_row(source_id)
    if source_row is None:
        return None
    return source_row.tree_index_json
```

Add `"tree_index_json"` to `_ALLOWED_UPDATE_FIELDS`.

**Step 5: Run all existing tests to verify no regressions**

Run: `pytest src/rfnry_rag/retrieval/tests/ -v`
Expected: PASS (existing tests unaffected)

**Step 6: Commit**

```bash
git add src/rfnry_rag/retrieval/stores/metadata/sqlalchemy.py src/rfnry_rag/retrieval/stores/metadata/base.py src/rfnry_rag/retrieval/tests/test_tree_metadata.py
git commit -m "feat(tree): add tree_index_json column to metadata store"
```

---

### Task 5: BAML Definitions — Indexing

Create BAML types and functions for tree indexing operations.

**Files:**
- Create: `src/rfnry_rag/retrieval/baml/baml_src/tree_indexing.baml`

**Step 1: Write the BAML definitions**

```baml
// Tree Indexing — BAML types and functions for building document tree indexes.

class TocDetectionResult {
    has_toc bool @description("whether the page contains a table of contents")
    has_page_numbers bool @description("whether the TOC entries include page numbers")
}

class TocEntry {
    structure string @description("hierarchical numbering, e.g. '1', '1.1', '1.1.2'")
    title string @description("section title text")
    page int | null @description("page number if present in TOC, null otherwise")
}

class TocStructure {
    entries TocEntry[] @description("ordered list of TOC entries with hierarchy")
}

class ExtractedSection {
    structure string @description("hierarchical numbering")
    title string @description("section title")
    start_page int @description("page where this section begins")
}

class ExtractedStructure {
    sections ExtractedSection[] @description("document sections discovered from content")
}

function DetectTableOfContents(page_text: string) -> TocDetectionResult {
    client TreeIndexingModel
    prompt #"
        Analyze the following page from a document. Determine if it contains
        a table of contents (TOC).

        PAGE TEXT:
        {{ page_text }}

        A table of contents typically lists section titles with optional page numbers.
        It may appear as a formatted list of chapters, sections, or headings.

        {{ ctx.output_format }}
    "#
}

function ParseTableOfContents(toc_text: string) -> TocStructure {
    client TreeIndexingModel
    prompt #"
        Parse this table of contents into a structured list of entries.
        Assign hierarchical structure codes (e.g., "1", "1.1", "1.1.2")
        based on the indentation, numbering, or visual hierarchy.

        TABLE OF CONTENTS:
        {{ toc_text }}

        Rules:
        - Preserve the original hierarchy (chapters > sections > subsections)
        - Include page numbers if they appear in the TOC
        - Set page to null if no page number is present for an entry
        - Use dot notation for structure: "1", "1.1", "1.2", "2", "2.1", etc.

        {{ ctx.output_format }}
    "#
}

function FindSectionStart(
    section_title: string,
    pages_text: string
) -> int {
    client TreeIndexingModel
    prompt #"
        Find the page number where the following section begins.

        SECTION TITLE: {{ section_title }}

        DOCUMENT PAGES:
        {{ pages_text }}

        Return the page number (as shown in the page headers) where this section
        title first appears as a heading. Return only the integer page number.

        {{ ctx.output_format }}
    "#
}

function VerifySectionPosition(title: string, page_text: string) -> bool {
    client TreeIndexingModel
    prompt #"
        Does the following section title appear as a heading on this page?

        SECTION TITLE: {{ title }}

        PAGE TEXT:
        {{ page_text }}

        Return true if the title (or a close match) appears as a section heading
        on this page. Return false otherwise.

        {{ ctx.output_format }}
    "#
}

function ExtractDocumentStructure(pages_text: string) -> ExtractedStructure {
    client TreeIndexingModel
    prompt #"
        This document does not have a table of contents. Analyze the content
        and extract the document structure by identifying sections and their
        starting pages.

        DOCUMENT PAGES:
        {{ pages_text }}

        Rules:
        - Identify section headings from formatting, numbering, or content shifts
        - Assign hierarchical structure codes ("1", "1.1", "1.2", "2", etc.)
        - Record the page number where each section begins
        - Only include significant sections, not every paragraph

        {{ ctx.output_format }}
    "#
}

function ContinueDocumentStructure(
    existing_structure: string,
    pages_text: string
) -> ExtractedStructure {
    client TreeIndexingModel
    prompt #"
        Continue extracting the document structure. The following sections
        have already been identified from earlier pages:

        EXISTING STRUCTURE:
        {{ existing_structure }}

        NEW PAGES:
        {{ pages_text }}

        Continue from where the existing structure left off. Use the same
        numbering scheme and hierarchy level. Only return NEW sections found
        in the new pages.

        {{ ctx.output_format }}
    "#
}

function GenerateNodeSummary(
    title: string,
    section_text: string
) -> string {
    client TreeIndexingModel
    prompt #"
        Write a concise summary (2-3 sentences) of this document section.
        Focus on the key topics, findings, or information covered.

        SECTION: {{ title }}

        TEXT:
        {{ section_text }}

        Return only the summary text, no formatting or labels.

        {{ ctx.output_format }}
    "#
}

function GenerateDocDescription(tree_structure: string) -> string {
    client TreeIndexingModel
    prompt #"
        Based on this document's structure, write a single sentence describing
        what this document is about.

        DOCUMENT STRUCTURE:
        {{ tree_structure }}

        Return only the one-sentence description.

        {{ ctx.output_format }}
    "#
}
```

**Step 2: Regenerate BAML client**

Run: `poe baml:generate:retrieval`
Expected: BAML client regenerated successfully

**Step 3: Commit**

```bash
git add src/rfnry_rag/retrieval/baml/baml_src/tree_indexing.baml src/rfnry_rag/retrieval/baml/baml_client/
git commit -m "feat(tree): add BAML definitions for tree indexing"
```

---

### Task 6: BAML Definitions — Search

Create BAML types and function for the tree search tool-use loop.

**Files:**
- Create: `src/rfnry_rag/retrieval/baml/baml_src/tree_search.baml`

**Step 1: Write the BAML definitions**

```baml
// Tree Search — BAML types and function for reasoning-based tree retrieval.

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

function TreeRetrievalStep(
    query: string,
    tree_structure: string,
    accumulated_context: string
) -> ToolFetchPages | ToolDrillDown | ToolResolvedPages {
    client TreeRetrievalModel
    prompt #"
        You are a document retrieval specialist. Given a query and a document's
        hierarchical structure, decide your next action to find the most relevant
        pages.

        QUERY: {{ query }}

        DOCUMENT STRUCTURE:
        {{ tree_structure }}

        {% if accumulated_context %}
        PAGES ALREADY RETRIEVED:
        {{ accumulated_context }}
        {% endif %}

        AVAILABLE ACTIONS:
        - ToolFetchPages: retrieve specific pages to read their content and
          gather evidence. Use this when you need to inspect page content
          before making a final decision.
        - ToolDrillDown: zoom into a specific subtree node to see its children
          in more detail. Use this when a section looks promising but you need
          to see its internal structure.
        - ToolResolvedPages: declare the final set of relevant pages. Use this
          when you have enough information to confidently identify which pages
          answer the query.

        Think step by step:
        1. Consider which sections are most likely relevant based on titles and summaries
        2. If you have already retrieved pages, assess whether they answer the query
        3. Choose the action that gets you closer to a confident answer

        {{ ctx.output_format }}
    "#
}
```

**Step 2: Regenerate BAML client**

Run: `poe baml:generate:retrieval`
Expected: BAML client regenerated successfully

**Step 3: Commit**

```bash
git add src/rfnry_rag/retrieval/baml/baml_src/tree_search.baml src/rfnry_rag/retrieval/baml/baml_client/
git commit -m "feat(tree): add BAML definitions for tree search tool-use loop"
```

---

### Task 7: Tree Indexing — TOC Detection & Parsing

Core logic for detecting and parsing document structure.

**Files:**
- Create: `src/rfnry_rag/retrieval/modules/ingestion/tree/__init__.py`
- Create: `src/rfnry_rag/retrieval/modules/ingestion/tree/toc.py`
- Test: `src/rfnry_rag/retrieval/modules/ingestion/tree/tests/__init__.py`
- Test: `src/rfnry_rag/retrieval/modules/ingestion/tree/tests/test_toc.py`

**Step 1: Write the failing tests**

```python
"""Tests for TOC detection and parsing."""

from unittest.mock import AsyncMock, patch

import pytest

from x64rag.retrieval.modules.ingestion.tree.toc import (
    PageContent,
    TocInfo,
    TocPath,
    detect_toc,
    find_section_starts,
    verify_section_positions,
)


def test_page_content_dataclass():
    page = PageContent(index=0, text="Hello world", token_count=2)
    assert page.index == 0
    assert page.text == "Hello world"


async def test_detect_toc_finds_toc_with_page_numbers():
    """When TOC is found with page numbers, return TocPath.WITH_PAGE_NUMBERS."""
    mock_result = AsyncMock()
    mock_result.has_toc = True
    mock_result.has_page_numbers = True

    pages = [PageContent(index=i, text=f"Page {i}", token_count=100) for i in range(5)]

    with patch(
        "x64rag.retrieval.modules.ingestion.tree.toc._detect_single_page",
        return_value=mock_result,
    ):
        info = await detect_toc(pages, toc_scan_pages=5)

    assert info.path == TocPath.WITH_PAGE_NUMBERS
    assert len(info.toc_pages) > 0


async def test_detect_toc_finds_toc_without_page_numbers():
    mock_result_toc = AsyncMock()
    mock_result_toc.has_toc = True
    mock_result_toc.has_page_numbers = False

    pages = [PageContent(index=i, text=f"Page {i}", token_count=100) for i in range(5)]

    with patch(
        "x64rag.retrieval.modules.ingestion.tree.toc._detect_single_page",
        return_value=mock_result_toc,
    ):
        info = await detect_toc(pages, toc_scan_pages=5)

    assert info.path == TocPath.WITHOUT_PAGE_NUMBERS


async def test_detect_toc_no_toc_found():
    mock_result = AsyncMock()
    mock_result.has_toc = False
    mock_result.has_page_numbers = False

    pages = [PageContent(index=i, text=f"Page {i}", token_count=100) for i in range(5)]

    with patch(
        "x64rag.retrieval.modules.ingestion.tree.toc._detect_single_page",
        return_value=mock_result,
    ):
        info = await detect_toc(pages, toc_scan_pages=5)

    assert info.path == TocPath.NO_TOC
    assert info.toc_pages == []


async def test_verify_section_positions():
    """Verify that section positions are checked concurrently."""
    sections = [
        {"title": "Chapter 1", "start_page": 5},
        {"title": "Chapter 2", "start_page": 15},
    ]
    pages = [PageContent(index=i, text=f"Page {i} content", token_count=100) for i in range(20)]

    with patch(
        "x64rag.retrieval.modules.ingestion.tree.toc._verify_single_position",
        return_value=True,
    ):
        results = await verify_section_positions(sections, pages)

    assert len(results) == 2
    assert all(r["verified"] for r in results)
```

**Step 2: Run test to verify it fails**

Run: `pytest src/rfnry_rag/retrieval/modules/ingestion/tree/tests/test_toc.py -v`
Expected: FAIL — module not found

**Step 3: Write the implementation**

Create `src/rfnry_rag/retrieval/modules/ingestion/tree/__init__.py` (empty).

Create `src/rfnry_rag/retrieval/modules/ingestion/tree/tests/__init__.py` (empty).

Create `src/rfnry_rag/retrieval/modules/ingestion/tree/toc.py`:

```python
"""TOC detection, parsing, and section position verification."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Any

from x64rag.retrieval.common.logging import get_logger

logger = get_logger(__name__)


class TocPath(Enum):
    WITH_PAGE_NUMBERS = "with_page_numbers"
    WITHOUT_PAGE_NUMBERS = "without_page_numbers"
    NO_TOC = "no_toc"


@dataclass
class PageContent:
    index: int
    text: str
    token_count: int


@dataclass
class TocInfo:
    path: TocPath
    toc_pages: list[PageContent]


async def _detect_single_page(page: PageContent, registry: Any) -> Any:
    from x64rag.retrieval.baml.baml_client.async_client import b

    return await b.DetectTableOfContents(page.text, {"client_registry": registry})


async def detect_toc(
    pages: list[PageContent],
    toc_scan_pages: int = 20,
    registry: Any = None,
) -> TocInfo:
    """Scan the first N pages for a table of contents."""
    scan_pages = pages[:toc_scan_pages]
    toc_pages: list[PageContent] = []

    for page in scan_pages:
        result = await _detect_single_page(page, registry)
        if result.has_toc:
            toc_pages.append(page)
            if result.has_page_numbers:
                return TocInfo(path=TocPath.WITH_PAGE_NUMBERS, toc_pages=toc_pages)

    if toc_pages:
        return TocInfo(path=TocPath.WITHOUT_PAGE_NUMBERS, toc_pages=toc_pages)

    return TocInfo(path=TocPath.NO_TOC, toc_pages=[])


async def parse_toc(toc_pages: list[PageContent], registry: Any = None) -> list[dict[str, Any]]:
    """Parse TOC pages into structured entries via BAML."""
    from x64rag.retrieval.baml.baml_client.async_client import b

    combined_text = "\n".join(p.text for p in toc_pages)
    result = await b.ParseTableOfContents(combined_text, {"client_registry": registry})

    return [
        {"structure": e.structure, "title": e.title, "page": e.page}
        for e in result.entries
    ]


async def find_section_starts(
    entries: list[dict[str, Any]],
    pages: list[PageContent],
    registry: Any = None,
) -> list[dict[str, Any]]:
    """For TOC entries without page numbers, find where each section starts."""
    from x64rag.retrieval.baml.baml_client.async_client import b

    pages_text = "\n\n".join(f"--- Page {p.index} ---\n{p.text}" for p in pages)
    results = []

    for entry in entries:
        page_num = await b.FindSectionStart(entry["title"], pages_text, {"client_registry": registry})
        results.append({**entry, "page": page_num})

    return results


async def _verify_single_position(
    title: str,
    page_text: str,
    registry: Any = None,
) -> bool:
    from x64rag.retrieval.baml.baml_client.async_client import b

    return await b.VerifySectionPosition(title, page_text, {"client_registry": registry})


async def verify_section_positions(
    sections: list[dict[str, Any]],
    pages: list[PageContent],
    registry: Any = None,
) -> list[dict[str, Any]]:
    """Concurrently verify that each section title appears on its indicated page."""
    page_by_index = {p.index: p for p in pages}
    tasks = []

    for section in sections:
        page_idx = section["start_page"]
        page = page_by_index.get(page_idx)
        if page:
            tasks.append(_verify_single_position(section["title"], page.text, registry))
        else:
            tasks.append(asyncio.coroutine(lambda: False)())

    results = await asyncio.gather(*tasks)
    return [
        {**section, "verified": verified}
        for section, verified in zip(sections, results, strict=True)
    ]
```

**Step 4: Run test to verify it passes**

Run: `pytest src/rfnry_rag/retrieval/modules/ingestion/tree/tests/test_toc.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/rfnry_rag/retrieval/modules/ingestion/tree/
git commit -m "feat(tree): add TOC detection and parsing"
```

---

### Task 8: Tree Indexing — Structure Building

Build the recursive tree from flat section lists, split large nodes, generate summaries.

**Files:**
- Create: `src/rfnry_rag/retrieval/modules/ingestion/tree/structure.py`
- Test: `src/rfnry_rag/retrieval/modules/ingestion/tree/tests/test_structure.py`

**Step 1: Write the failing tests**

```python
"""Tests for tree structure building."""

from unittest.mock import AsyncMock, patch

import pytest

from x64rag.retrieval.common.models import TreeNode
from x64rag.retrieval.modules.ingestion.tree.structure import (
    build_tree,
    calculate_page_ranges,
    split_large_nodes,
)
from x64rag.retrieval.modules.ingestion.tree.toc import PageContent


def test_build_tree_flat_sections():
    """Top-level sections with no nesting."""
    sections = [
        {"structure": "1", "title": "Introduction", "page": 1},
        {"structure": "2", "title": "Methods", "page": 10},
        {"structure": "3", "title": "Results", "page": 20},
    ]
    nodes = build_tree(sections)
    assert len(nodes) == 3
    assert nodes[0].title == "Introduction"
    assert nodes[1].title == "Methods"
    assert all(n.children == [] for n in nodes)


def test_build_tree_nested_sections():
    """Sections with parent-child relationships."""
    sections = [
        {"structure": "1", "title": "Chapter 1", "page": 1},
        {"structure": "1.1", "title": "Section 1.1", "page": 3},
        {"structure": "1.2", "title": "Section 1.2", "page": 7},
        {"structure": "2", "title": "Chapter 2", "page": 15},
    ]
    nodes = build_tree(sections)
    assert len(nodes) == 2
    assert len(nodes[0].children) == 2
    assert nodes[0].children[0].title == "Section 1.1"
    assert nodes[0].children[1].title == "Section 1.2"
    assert nodes[1].children == []


def test_build_tree_deep_nesting():
    sections = [
        {"structure": "1", "title": "A", "page": 1},
        {"structure": "1.1", "title": "B", "page": 2},
        {"structure": "1.1.1", "title": "C", "page": 3},
    ]
    nodes = build_tree(sections)
    assert len(nodes) == 1
    assert len(nodes[0].children) == 1
    assert len(nodes[0].children[0].children) == 1
    assert nodes[0].children[0].children[0].title == "C"


def test_calculate_page_ranges():
    """End index should be set based on next sibling or parent."""
    child1 = TreeNode(node_id="0002", title="S1.1", start_index=1, end_index=0)
    child2 = TreeNode(node_id="0003", title="S1.2", start_index=5, end_index=0)
    root = TreeNode(
        node_id="0001", title="Ch1", start_index=1, end_index=0, children=[child1, child2]
    )
    calculate_page_ranges([root], total_pages=20)
    assert root.end_index == 20
    assert child1.end_index == 4
    assert child2.end_index == 20


def test_calculate_page_ranges_multiple_roots():
    root1 = TreeNode(node_id="0001", title="Ch1", start_index=1, end_index=0)
    root2 = TreeNode(node_id="0002", title="Ch2", start_index=10, end_index=0)
    calculate_page_ranges([root1, root2], total_pages=25)
    assert root1.end_index == 9
    assert root2.end_index == 25


def test_split_large_nodes_no_split_needed():
    pages = [PageContent(index=i, text=f"Page {i}", token_count=100) for i in range(5)]
    node = TreeNode(node_id="0001", title="Small", start_index=0, end_index=4)
    result = split_large_nodes([node], pages, max_pages=10, max_tokens=20_000)
    assert len(result) == 1
    assert result[0].children == []
```

**Step 2: Run test to verify it fails**

Run: `pytest src/rfnry_rag/retrieval/modules/ingestion/tree/tests/test_structure.py -v`
Expected: FAIL — module not found

**Step 3: Write the implementation**

Create `src/rfnry_rag/retrieval/modules/ingestion/tree/structure.py`:

```python
"""Tree structure building, page range calculation, and large node splitting."""

from __future__ import annotations

from typing import Any

from x64rag.retrieval.common.models import TreeNode
from x64rag.retrieval.modules.ingestion.tree.toc import PageContent

_node_counter = 0


def _next_node_id() -> str:
    global _node_counter
    _node_counter += 1
    return f"{_node_counter:04d}"


def reset_node_counter() -> None:
    global _node_counter
    _node_counter = 0


def build_tree(sections: list[dict[str, Any]]) -> list[TreeNode]:
    """Convert flat section list with structure codes into recursive TreeNode hierarchy.

    Structure codes use dot notation: "1", "1.1", "1.2", "2", "2.1", etc.
    """
    reset_node_counter()
    nodes_by_structure: dict[str, TreeNode] = {}
    roots: list[TreeNode] = []

    for section in sections:
        structure = section["structure"]
        node = TreeNode(
            node_id=_next_node_id(),
            title=section["title"],
            start_index=section.get("page", 0),
            end_index=0,
        )
        nodes_by_structure[structure] = node

        parts = structure.rsplit(".", 1)
        if len(parts) == 1:
            roots.append(node)
        else:
            parent_structure = parts[0]
            parent = nodes_by_structure.get(parent_structure)
            if parent:
                parent.children.append(node)
            else:
                roots.append(node)

    return roots


def calculate_page_ranges(nodes: list[TreeNode], total_pages: int) -> None:
    """Calculate end_index for each node based on sibling boundaries."""
    for i, node in enumerate(nodes):
        if i + 1 < len(nodes):
            node.end_index = nodes[i + 1].start_index - 1
        else:
            node.end_index = total_pages

        if node.children:
            calculate_page_ranges(node.children, node.end_index)


def _node_page_count(node: TreeNode) -> int:
    return node.end_index - node.start_index + 1


def _node_token_count(node: TreeNode, pages: list[PageContent]) -> int:
    page_by_index = {p.index: p for p in pages}
    total = 0
    for idx in range(node.start_index, node.end_index + 1):
        page = page_by_index.get(idx)
        if page:
            total += page.token_count
    return total


def split_large_nodes(
    nodes: list[TreeNode],
    pages: list[PageContent],
    max_pages: int = 10,
    max_tokens: int = 20_000,
) -> list[TreeNode]:
    """Recursively split nodes that exceed max_pages AND max_tokens."""
    for node in nodes:
        if node.children:
            node.children = split_large_nodes(node.children, pages, max_pages, max_tokens)
        elif (
            _node_page_count(node) > max_pages
            and _node_token_count(node, pages) > max_tokens
        ):
            midpoint = (node.start_index + node.end_index) // 2
            left = TreeNode(
                node_id=_next_node_id(),
                title=f"{node.title} (Part 1)",
                start_index=node.start_index,
                end_index=midpoint,
            )
            right = TreeNode(
                node_id=_next_node_id(),
                title=f"{node.title} (Part 2)",
                start_index=midpoint + 1,
                end_index=node.end_index,
            )
            node.children = split_large_nodes([left, right], pages, max_pages, max_tokens)

    return nodes
```

**Step 4: Run test to verify it passes**

Run: `pytest src/rfnry_rag/retrieval/modules/ingestion/tree/tests/test_structure.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/rfnry_rag/retrieval/modules/ingestion/tree/structure.py src/rfnry_rag/retrieval/modules/ingestion/tree/tests/test_structure.py
git commit -m "feat(tree): add tree structure building and page range calculation"
```

---

### Task 9: Tree Indexing — Service

Orchestrate the full indexing pipeline.

**Files:**
- Create: `src/rfnry_rag/retrieval/modules/ingestion/tree/service.py`
- Test: `src/rfnry_rag/retrieval/modules/ingestion/tree/tests/test_service.py`

**Step 1: Write the failing tests**

```python
"""Tests for TreeIndexingService."""

import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from x64rag.retrieval.common.errors import TreeIndexingError
from x64rag.retrieval.common.models import TreeIndex
from x64rag.retrieval.modules.ingestion.tree.service import TreeIndexingService
from x64rag.retrieval.modules.ingestion.tree.toc import PageContent, TocInfo, TocPath
from x64rag.retrieval.server import TreeIndexingConfig


def _make_service(config: TreeIndexingConfig | None = None) -> TreeIndexingService:
    return TreeIndexingService(
        config=config or TreeIndexingConfig(enabled=True),
        metadata_store=AsyncMock(),
        registry=MagicMock(),
    )


async def test_index_pdf_with_toc():
    service = _make_service()
    pages = [PageContent(index=i, text=f"Page {i} content", token_count=500) for i in range(20)]
    toc_info = TocInfo(path=TocPath.WITH_PAGE_NUMBERS, toc_pages=[pages[0]])

    parsed_entries = [
        {"structure": "1", "title": "Introduction", "page": 1},
        {"structure": "2", "title": "Methods", "page": 10},
    ]

    with (
        patch(
            "x64rag.retrieval.modules.ingestion.tree.service.detect_toc",
            return_value=toc_info,
        ),
        patch(
            "x64rag.retrieval.modules.ingestion.tree.service.parse_toc",
            return_value=parsed_entries,
        ),
        patch(
            "x64rag.retrieval.modules.ingestion.tree.service.verify_section_positions",
            return_value=[{**e, "verified": True} for e in parsed_entries],
        ),
        patch(
            "x64rag.retrieval.modules.ingestion.tree.service._generate_summaries",
            return_value=None,
        ),
        patch(
            "x64rag.retrieval.modules.ingestion.tree.service._generate_doc_description",
            return_value="A document about methods.",
        ),
    ):
        tree_index = await service.build_tree_index(
            source_id="src-1",
            doc_name="test.pdf",
            pages=pages,
        )

    assert isinstance(tree_index, TreeIndex)
    assert tree_index.source_id == "src-1"
    assert len(tree_index.structure) == 2
    assert tree_index.structure[0].title == "Introduction"


async def test_index_pdf_no_toc():
    service = _make_service()
    pages = [PageContent(index=i, text=f"Page {i} content", token_count=500) for i in range(10)]
    toc_info = TocInfo(path=TocPath.NO_TOC, toc_pages=[])

    extracted = [
        {"structure": "1", "title": "Overview", "start_page": 0},
        {"structure": "2", "title": "Details", "start_page": 5},
    ]

    with (
        patch(
            "x64rag.retrieval.modules.ingestion.tree.service.detect_toc",
            return_value=toc_info,
        ),
        patch(
            "x64rag.retrieval.modules.ingestion.tree.service._extract_structure_no_toc",
            return_value=[{"structure": s["structure"], "title": s["title"], "page": s["start_page"]} for s in extracted],
        ),
        patch(
            "x64rag.retrieval.modules.ingestion.tree.service.verify_section_positions",
            return_value=[{**e, "verified": True, "start_page": e["page"]} for e in [{"structure": s["structure"], "title": s["title"], "page": s["start_page"]} for s in extracted]],
        ),
        patch(
            "x64rag.retrieval.modules.ingestion.tree.service._generate_summaries",
            return_value=None,
        ),
        patch(
            "x64rag.retrieval.modules.ingestion.tree.service._generate_doc_description",
            return_value=None,
        ),
    ):
        tree_index = await service.build_tree_index(
            source_id="src-1",
            doc_name="test.pdf",
            pages=pages,
        )

    assert isinstance(tree_index, TreeIndex)
    assert len(tree_index.structure) == 2


async def test_index_saves_to_metadata_store():
    service = _make_service()
    pages = [PageContent(index=i, text=f"Page {i}", token_count=100) for i in range(5)]
    toc_info = TocInfo(path=TocPath.NO_TOC, toc_pages=[])

    with (
        patch(
            "x64rag.retrieval.modules.ingestion.tree.service.detect_toc",
            return_value=toc_info,
        ),
        patch(
            "x64rag.retrieval.modules.ingestion.tree.service._extract_structure_no_toc",
            return_value=[{"structure": "1", "title": "All", "page": 0}],
        ),
        patch(
            "x64rag.retrieval.modules.ingestion.tree.service.verify_section_positions",
            return_value=[{"structure": "1", "title": "All", "page": 0, "start_page": 0, "verified": True}],
        ),
        patch(
            "x64rag.retrieval.modules.ingestion.tree.service._generate_summaries",
            return_value=None,
        ),
        patch(
            "x64rag.retrieval.modules.ingestion.tree.service._generate_doc_description",
            return_value=None,
        ),
    ):
        tree_index = await service.build_tree_index(
            source_id="src-1",
            doc_name="test.pdf",
            pages=pages,
        )
        await service.save_tree_index(tree_index)

    service._metadata_store.save_tree_index.assert_awaited_once()
    call_args = service._metadata_store.save_tree_index.call_args
    assert call_args[0][0] == "src-1"
    saved_json = json.loads(call_args[0][1])
    assert saved_json["source_id"] == "src-1"
```

**Step 2: Run test to verify it fails**

Run: `pytest src/rfnry_rag/retrieval/modules/ingestion/tree/tests/test_service.py -v`
Expected: FAIL — module not found

**Step 3: Write the implementation**

Create `src/rfnry_rag/retrieval/modules/ingestion/tree/service.py`:

```python
"""TreeIndexingService — orchestrates the full tree indexing pipeline."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from x64rag.common.concurrency import run_concurrent
from x64rag.retrieval.common.errors import TreeIndexingError
from x64rag.retrieval.common.logging import get_logger
from x64rag.retrieval.common.models import TreeIndex
from x64rag.retrieval.modules.ingestion.tree.structure import (
    build_tree,
    calculate_page_ranges,
    split_large_nodes,
)
from x64rag.retrieval.modules.ingestion.tree.toc import (
    PageContent,
    TocPath,
    detect_toc,
    find_section_starts,
    parse_toc,
    verify_section_positions,
)
from x64rag.retrieval.server import TreeIndexingConfig
from x64rag.retrieval.stores.metadata.base import BaseMetadataStore

logger = get_logger(__name__)


async def _extract_structure_no_toc(
    pages: list[PageContent],
    registry: Any = None,
) -> list[dict[str, Any]]:
    """Extract document structure when no TOC is found, using LLM."""
    from x64rag.retrieval.baml.baml_client.async_client import b

    # Process pages in groups that fit context windows
    group_size = 20
    all_sections: list[dict[str, Any]] = []

    for start in range(0, len(pages), group_size):
        group = pages[start : start + group_size]
        pages_text = "\n\n".join(f"--- Page {p.index} ---\n{p.text}" for p in group)

        if not all_sections:
            result = await b.ExtractDocumentStructure(pages_text, {"client_registry": registry})
        else:
            existing = json.dumps(all_sections)
            result = await b.ContinueDocumentStructure(existing, pages_text, {"client_registry": registry})

        for s in result.sections:
            all_sections.append({
                "structure": s.structure,
                "title": s.title,
                "page": s.start_page,
            })

    return all_sections


async def _generate_summaries(
    nodes: list[Any],
    pages: list[PageContent],
    registry: Any = None,
) -> None:
    """Generate summaries for all nodes concurrently."""
    from x64rag.retrieval.baml.baml_client.async_client import b

    page_by_index = {p.index: p for p in pages}

    async def _summarize_node(node: Any) -> None:
        text_parts = []
        for idx in range(node.start_index, node.end_index + 1):
            page = page_by_index.get(idx)
            if page:
                text_parts.append(page.text)
        if text_parts:
            section_text = "\n\n".join(text_parts)
            node.summary = await b.GenerateNodeSummary(
                node.title, section_text, {"client_registry": registry}
            )
        if node.children:
            await _generate_summaries(node.children, pages, registry)

    await run_concurrent(nodes, _summarize_node, concurrency=10)


async def _generate_doc_description(
    tree_structure: str,
    registry: Any = None,
) -> str | None:
    """Generate a one-sentence document description."""
    from x64rag.retrieval.baml.baml_client.async_client import b

    return await b.GenerateDocDescription(tree_structure, {"client_registry": registry})


class TreeIndexingService:
    """Orchestrates the tree indexing pipeline."""

    def __init__(
        self,
        config: TreeIndexingConfig,
        metadata_store: BaseMetadataStore,
        registry: Any = None,
    ) -> None:
        self._config = config
        self._metadata_store = metadata_store
        self._registry = registry

    async def build_tree_index(
        self,
        source_id: str,
        doc_name: str,
        pages: list[PageContent],
    ) -> TreeIndex:
        """Build a complete tree index from document pages."""
        logger.info("Building tree index for %s (%d pages)", doc_name, len(pages))

        # Step 1: Detect TOC
        toc_info = await detect_toc(pages, self._config.toc_scan_pages, self._registry)
        logger.info("TOC detection result: %s", toc_info.path.value)

        # Step 2: Get section list based on TOC path
        if toc_info.path == TocPath.WITH_PAGE_NUMBERS:
            sections = await parse_toc(toc_info.toc_pages, self._registry)
        elif toc_info.path == TocPath.WITHOUT_PAGE_NUMBERS:
            sections = await parse_toc(toc_info.toc_pages, self._registry)
            sections = await find_section_starts(sections, pages, self._registry)
        else:
            sections = await _extract_structure_no_toc(pages, self._registry)

        if not sections:
            raise TreeIndexingError(f"No document structure found for {doc_name}")

        # Step 3: Verify section positions
        verified = await verify_section_positions(
            [{"title": s["title"], "start_page": s["page"]} for s in sections],
            pages,
            self._registry,
        )
        logger.info(
            "Position verification: %d/%d confirmed",
            sum(1 for v in verified if v.get("verified")),
            len(verified),
        )

        # Step 4: Build tree
        nodes = build_tree(sections)
        calculate_page_ranges(nodes, len(pages) - 1)

        # Step 5: Split large nodes
        nodes = split_large_nodes(
            nodes,
            pages,
            max_pages=self._config.max_pages_per_node,
            max_tokens=self._config.max_tokens_per_node,
        )

        # Step 6: Generate summaries
        if self._config.generate_summaries:
            await _generate_summaries(nodes, pages, self._registry)

        # Step 7: Generate doc description
        doc_description = None
        if self._config.generate_description:
            tree_str = json.dumps([n.to_dict() for n in nodes], indent=2)
            doc_description = await _generate_doc_description(tree_str, self._registry)

        tree_index = TreeIndex(
            source_id=source_id,
            doc_name=doc_name,
            doc_description=doc_description,
            structure=nodes,
            page_count=len(pages),
            created_at=datetime.now(timezone.utc),
        )

        logger.info("Tree index built: %d root nodes", len(nodes))
        return tree_index

    async def save_tree_index(self, tree_index: TreeIndex) -> None:
        """Persist the tree index to the metadata store."""
        tree_json = json.dumps(tree_index.to_dict())
        await self._metadata_store.save_tree_index(tree_index.source_id, tree_json)
        logger.info("Tree index saved for source %s", tree_index.source_id)
```

**Step 4: Run test to verify it passes**

Run: `pytest src/rfnry_rag/retrieval/modules/ingestion/tree/tests/test_service.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/rfnry_rag/retrieval/modules/ingestion/tree/service.py src/rfnry_rag/retrieval/modules/ingestion/tree/tests/test_service.py
git commit -m "feat(tree): add TreeIndexingService"
```

---

### Task 10: Tree Search — Tool Handlers

Implement the Python-side tool execution for the BAML loop.

**Files:**
- Create: `src/rfnry_rag/retrieval/modules/retrieval/tree/__init__.py`
- Create: `src/rfnry_rag/retrieval/modules/retrieval/tree/tools.py`
- Test: `src/rfnry_rag/retrieval/modules/retrieval/tree/tests/__init__.py`
- Test: `src/rfnry_rag/retrieval/modules/retrieval/tree/tests/test_tools.py`

**Step 1: Write the failing tests**

```python
"""Tests for tree search tool handlers."""

from x64rag.retrieval.common.models import TreeIndex, TreeNode
from x64rag.retrieval.modules.ingestion.tree.toc import PageContent
from x64rag.retrieval.modules.retrieval.tree.tools import (
    fetch_pages,
    get_subtree,
    parse_page_ranges,
    serialize_tree_for_prompt,
)


def test_parse_page_ranges_single():
    assert parse_page_ranges("5") == [5]


def test_parse_page_ranges_range():
    assert parse_page_ranges("5-7") == [5, 6, 7]


def test_parse_page_ranges_mixed():
    assert parse_page_ranges("3,5-7,12") == [3, 5, 6, 7, 12]


def test_parse_page_ranges_whitespace():
    assert parse_page_ranges(" 3 , 5 - 7 ") == [3, 5, 6, 7]


def test_fetch_pages():
    pages = [PageContent(index=i, text=f"Content of page {i}", token_count=100) for i in range(10)]
    result = fetch_pages("2-4", pages)
    assert "Content of page 2" in result
    assert "Content of page 3" in result
    assert "Content of page 4" in result
    assert "Content of page 5" not in result


def test_get_subtree():
    child = TreeNode(node_id="0002", title="Section 1.1", start_index=1, end_index=5)
    grandchild = TreeNode(node_id="0003", title="Sub 1.1.1", start_index=1, end_index=3)
    child.children = [grandchild]
    root = TreeNode(node_id="0001", title="Chapter 1", start_index=1, end_index=10, children=[child])

    subtree = get_subtree([root], "0002")
    assert subtree is not None
    assert subtree.node_id == "0002"
    assert len(subtree.children) == 1


def test_get_subtree_not_found():
    root = TreeNode(node_id="0001", title="Chapter 1", start_index=1, end_index=10)
    assert get_subtree([root], "9999") is None


def test_serialize_tree_for_prompt():
    child = TreeNode(
        node_id="0002", title="Section 1.1", start_index=1, end_index=5, summary="Details"
    )
    root = TreeNode(
        node_id="0001", title="Chapter 1", start_index=1, end_index=10,
        summary="Overview", children=[child],
    )
    result = serialize_tree_for_prompt([root])
    assert "0001" in result
    assert "Chapter 1" in result
    assert "pages 1-10" in result
    assert "Overview" in result
    assert "Section 1.1" in result


def test_serialize_tree_for_prompt_no_summary():
    root = TreeNode(node_id="0001", title="Chapter 1", start_index=1, end_index=10)
    result = serialize_tree_for_prompt([root])
    assert "0001" in result
    assert "Chapter 1" in result
```

**Step 2: Run test to verify it fails**

Run: `pytest src/rfnry_rag/retrieval/modules/retrieval/tree/tests/test_tools.py -v`
Expected: FAIL — module not found

**Step 3: Write the implementation**

Create `src/rfnry_rag/retrieval/modules/retrieval/tree/__init__.py` (empty).

Create `src/rfnry_rag/retrieval/modules/retrieval/tree/tests/__init__.py` (empty).

Create `src/rfnry_rag/retrieval/modules/retrieval/tree/tools.py`:

```python
"""Tool execution handlers for tree search."""

from __future__ import annotations

from x64rag.retrieval.common.models import TreeNode
from x64rag.retrieval.modules.ingestion.tree.toc import PageContent


def parse_page_ranges(pages_str: str) -> list[int]:
    """Parse page range string like '3,5-7,12' into a sorted list of page indices."""
    result: list[int] = []
    for part in pages_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            result.extend(range(int(start.strip()), int(end.strip()) + 1))
        else:
            result.append(int(part))
    return sorted(result)


def fetch_pages(pages_str: str, pages: list[PageContent]) -> str:
    """Fetch page content for the given page ranges."""
    indices = set(parse_page_ranges(pages_str))
    page_by_index = {p.index: p for p in pages}
    parts: list[str] = []

    for idx in sorted(indices):
        page = page_by_index.get(idx)
        if page:
            parts.append(f"--- Page {idx} ---\n{page.text}")

    return "\n\n".join(parts)


def get_subtree(nodes: list[TreeNode], node_id: str) -> TreeNode | None:
    """Find a node by ID in the tree."""
    for node in nodes:
        if node.node_id == node_id:
            return node
        if node.children:
            found = get_subtree(node.children, node_id)
            if found:
                return found
    return None


def serialize_tree_for_prompt(nodes: list[TreeNode], indent: int = 0) -> str:
    """Serialize tree structure into a compact text format for LLM prompts."""
    lines: list[str] = []
    prefix = "  " * indent

    for node in nodes:
        line = f"{prefix}[{node.node_id}] {node.title} (pages {node.start_index}-{node.end_index})"
        if node.summary:
            line += f"\n{prefix}  Summary: {node.summary}"
        lines.append(line)

        if node.children:
            lines.append(serialize_tree_for_prompt(node.children, indent + 1))

    return "\n".join(lines)
```

**Step 4: Run test to verify it passes**

Run: `pytest src/rfnry_rag/retrieval/modules/retrieval/tree/tests/test_tools.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/rfnry_rag/retrieval/modules/retrieval/tree/
git commit -m "feat(tree): add tree search tool handlers"
```

---

### Task 11: Tree Search — Service

The BAML tool-use loop that orchestrates tree retrieval.

**Files:**
- Create: `src/rfnry_rag/retrieval/modules/retrieval/tree/service.py`
- Test: `src/rfnry_rag/retrieval/modules/retrieval/tree/tests/test_search_service.py`

**Step 1: Write the failing tests**

```python
"""Tests for TreeSearchService."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from x64rag.retrieval.common.errors import TreeSearchError
from x64rag.retrieval.common.models import RetrievedChunk, TreeIndex, TreeNode, TreeSearchResult
from x64rag.retrieval.modules.ingestion.tree.toc import PageContent
from x64rag.retrieval.modules.retrieval.tree.service import TreeSearchService
from x64rag.retrieval.server import TreeSearchConfig


def _make_tree_index() -> TreeIndex:
    from datetime import datetime, timezone

    child = TreeNode(node_id="0002", title="Methods", start_index=5, end_index=10, summary="How we did it")
    root = TreeNode(
        node_id="0001", title="Report", start_index=0, end_index=15,
        summary="Full report", children=[child],
    )
    return TreeIndex(
        source_id="src-1",
        doc_name="report.pdf",
        doc_description="Annual report",
        structure=[root],
        page_count=16,
        created_at=datetime.now(timezone.utc),
    )


def _make_pages(count: int = 16) -> list[PageContent]:
    return [PageContent(index=i, text=f"Content of page {i}", token_count=200) for i in range(count)]


def _make_service(config: TreeSearchConfig | None = None) -> TreeSearchService:
    return TreeSearchService(
        config=config or TreeSearchConfig(enabled=True),
        registry=MagicMock(),
    )


async def test_search_resolves_in_one_step():
    """When LLM immediately returns ToolResolvedPages, search completes in one step."""
    service = _make_service()
    tree_index = _make_tree_index()
    pages = _make_pages()

    resolved = SimpleNamespace(pages="5-7", reasoning="Methods section answers the query")

    with patch(
        "x64rag.retrieval.modules.retrieval.tree.service._call_retrieval_step",
        return_value=resolved,
    ):
        results = await service.search("What methods were used?", tree_index, pages)

    assert len(results) == 1
    assert results[0].pages == "5-7"
    assert "Methods" in results[0].reasoning or "methods" in results[0].reasoning.lower()


async def test_search_fetch_then_resolve():
    """LLM fetches pages first, then resolves."""
    service = _make_service()
    tree_index = _make_tree_index()
    pages = _make_pages()

    fetch_action = SimpleNamespace(pages="5-7", reasoning="Need to check methods")
    resolved = SimpleNamespace(pages="5-6", reasoning="Found the answer")

    call_count = 0

    async def mock_step(query, tree_str, context, registry):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return fetch_action
        return resolved

    with patch(
        "x64rag.retrieval.modules.retrieval.tree.service._call_retrieval_step",
        side_effect=mock_step,
    ):
        results = await service.search("What methods?", tree_index, pages)

    assert call_count == 2
    assert results[0].pages == "5-6"


async def test_search_max_steps_exceeded():
    """When max_steps is hit, return whatever pages have been fetched."""
    service = _make_service(TreeSearchConfig(enabled=True, max_steps=2))
    tree_index = _make_tree_index()
    pages = _make_pages()

    fetch_action = SimpleNamespace(pages="5-7", reasoning="Checking")

    with patch(
        "x64rag.retrieval.modules.retrieval.tree.service._call_retrieval_step",
        return_value=fetch_action,
    ):
        results = await service.search("What?", tree_index, pages)

    # Should return results from fetched pages even without explicit resolution
    assert len(results) >= 1


async def test_search_converts_to_retrieved_chunks():
    """Tree search results can be converted to RetrievedChunk format."""
    service = _make_service()
    tree_index = _make_tree_index()
    pages = _make_pages()

    resolved = SimpleNamespace(pages="5-7", reasoning="Methods section")

    with patch(
        "x64rag.retrieval.modules.retrieval.tree.service._call_retrieval_step",
        return_value=resolved,
    ):
        results = await service.search("What methods?", tree_index, pages)
        chunks = service.to_retrieved_chunks(results, tree_index)

    assert all(isinstance(c, RetrievedChunk) for c in chunks)
    assert chunks[0].source_id == "src-1"
```

**Step 2: Run test to verify it fails**

Run: `pytest src/rfnry_rag/retrieval/modules/retrieval/tree/tests/test_search_service.py -v`
Expected: FAIL — module not found

**Step 3: Write the implementation**

Create `src/rfnry_rag/retrieval/modules/retrieval/tree/service.py`:

```python
"""TreeSearchService — BAML tool-use loop for tree-based retrieval."""

from __future__ import annotations

from typing import Any

from x64rag.retrieval.common.errors import TreeSearchError
from x64rag.retrieval.common.logging import get_logger
from x64rag.retrieval.common.models import RetrievedChunk, TreeIndex, TreeSearchResult
from x64rag.retrieval.modules.ingestion.tree.toc import PageContent
from x64rag.retrieval.modules.retrieval.tree.tools import (
    fetch_pages,
    get_subtree,
    parse_page_ranges,
    serialize_tree_for_prompt,
)
from x64rag.retrieval.server import TreeSearchConfig

logger = get_logger(__name__)

# Import BAML types for isinstance checks
_TOOL_TYPES_LOADED = False
_ToolFetchPages: type | None = None
_ToolDrillDown: type | None = None
_ToolResolvedPages: type | None = None


def _load_tool_types() -> None:
    global _TOOL_TYPES_LOADED, _ToolFetchPages, _ToolDrillDown, _ToolResolvedPages
    if _TOOL_TYPES_LOADED:
        return
    from x64rag.retrieval.baml.baml_client.types import (
        ToolDrillDown,
        ToolFetchPages,
        ToolResolvedPages,
    )
    _ToolFetchPages = ToolFetchPages
    _ToolDrillDown = ToolDrillDown
    _ToolResolvedPages = ToolResolvedPages
    _TOOL_TYPES_LOADED = True


async def _call_retrieval_step(
    query: str,
    tree_structure: str,
    accumulated_context: str,
    registry: Any = None,
) -> Any:
    from x64rag.retrieval.baml.baml_client.async_client import b

    return await b.TreeRetrievalStep(
        query, tree_structure, accumulated_context, {"client_registry": registry}
    )


class TreeSearchService:
    """Orchestrates tree-based retrieval using a BAML tool-use loop."""

    def __init__(
        self,
        config: TreeSearchConfig,
        registry: Any = None,
    ) -> None:
        self._config = config
        self._registry = registry

    async def search(
        self,
        query: str,
        tree_index: TreeIndex,
        pages: list[PageContent],
    ) -> list[TreeSearchResult]:
        """Run the tree search loop and return results."""
        _load_tool_types()

        tree_nodes = tree_index.structure
        tree_str = serialize_tree_for_prompt(tree_nodes)
        accumulated_context = ""
        fetched_pages: set[int] = set()

        for step in range(self._config.max_steps):
            logger.info("Tree search step %d/%d", step + 1, self._config.max_steps)

            action = await _call_retrieval_step(
                query, tree_str, accumulated_context, self._registry
            )

            if _ToolResolvedPages and isinstance(action, _ToolResolvedPages):
                logger.info("Tree search resolved: %s", action.pages)
                page_indices = parse_page_ranges(action.pages)
                content = fetch_pages(action.pages, pages)
                return [
                    TreeSearchResult(
                        node_id="resolved",
                        title="Tree Search Result",
                        pages=action.pages,
                        content=content,
                        reasoning=action.reasoning,
                    )
                ]

            if _ToolFetchPages and isinstance(action, _ToolFetchPages):
                logger.info("Fetching pages: %s", action.pages)
                new_content = fetch_pages(action.pages, pages)
                accumulated_context += f"\n\n{new_content}"
                fetched_pages.update(parse_page_ranges(action.pages))

                # Check context token limit
                if len(accumulated_context) > self._config.max_context_tokens:
                    logger.warning("Max context tokens exceeded, resolving with fetched pages")
                    break

            elif _ToolDrillDown and isinstance(action, _ToolDrillDown):
                logger.info("Drilling down into node: %s", action.node_id)
                subtree = get_subtree(tree_nodes, action.node_id)
                if subtree and subtree.children:
                    tree_str = serialize_tree_for_prompt(subtree.children)
                else:
                    logger.warning("DrillDown target %s has no children", action.node_id)

        # Max steps exceeded or context limit hit — return fetched pages
        if fetched_pages:
            pages_str = ",".join(str(p) for p in sorted(fetched_pages))
            content = fetch_pages(pages_str, pages)
            return [
                TreeSearchResult(
                    node_id="fallback",
                    title="Tree Search Result (max steps)",
                    pages=pages_str,
                    content=content,
                    reasoning="Search reached step limit; returning all fetched pages.",
                )
            ]

        return []

    @staticmethod
    def to_retrieved_chunks(
        results: list[TreeSearchResult],
        tree_index: TreeIndex,
    ) -> list[RetrievedChunk]:
        """Convert tree search results to RetrievedChunk format for RRF fusion."""
        chunks: list[RetrievedChunk] = []
        for i, result in enumerate(results):
            chunks.append(
                RetrievedChunk(
                    chunk_id=f"tree-{result.node_id}-{i}",
                    source_id=tree_index.source_id,
                    content=result.content,
                    score=1.0,  # tree results are pre-filtered by relevance
                    source_metadata={
                        "name": tree_index.doc_name,
                        "tree_pages": result.pages,
                        "tree_reasoning": result.reasoning,
                    },
                )
            )
        return chunks
```

**Step 4: Run test to verify it passes**

Run: `pytest src/rfnry_rag/retrieval/modules/retrieval/tree/tests/test_search_service.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/rfnry_rag/retrieval/modules/retrieval/tree/
git commit -m "feat(tree): add TreeSearchService with BAML tool-use loop"
```

---

### Task 12: Integration — RetrievalService

Add tree search as a named task in the existing multi-path retrieval.

**Files:**
- Modify: `src/rfnry_rag/retrieval/modules/retrieval/search/service.py`
- Test: `src/rfnry_rag/retrieval/modules/retrieval/search/tests/test_tree_integration.py`

**Step 1: Write the failing test**

```python
"""Tests for tree search integration in RetrievalService."""

from unittest.mock import AsyncMock, MagicMock

from x64rag.retrieval.common.models import RetrievedChunk
from x64rag.retrieval.modules.retrieval.search.service import RetrievalService


async def test_retrieval_service_includes_tree_results():
    """When tree_search is provided, its results are included in retrieval."""
    vector_search = AsyncMock()
    vector_search.search = AsyncMock(
        return_value=[
            RetrievedChunk(chunk_id="v1", source_id="s1", content="vector result", score=0.9)
        ]
    )

    tree_search = AsyncMock()
    tree_search.search = AsyncMock(
        return_value=[
            RetrievedChunk(chunk_id="t1", source_id="s1", content="tree result", score=1.0)
        ]
    )

    service = RetrievalService(
        vector_search=vector_search,
        tree_search=tree_search,
    )

    results = await service.retrieve("test query")
    chunk_ids = {r.chunk_id for r in results}
    assert "v1" in chunk_ids
    assert "t1" in chunk_ids


async def test_retrieval_service_without_tree():
    """When no tree_search, retrieval works as before."""
    vector_search = AsyncMock()
    vector_search.search = AsyncMock(
        return_value=[
            RetrievedChunk(chunk_id="v1", source_id="s1", content="vector result", score=0.9)
        ]
    )

    service = RetrievalService(vector_search=vector_search)
    results = await service.retrieve("test query")
    assert len(results) == 1
    assert results[0].chunk_id == "v1"
```

**Step 2: Run test to verify it fails**

Run: `pytest src/rfnry_rag/retrieval/modules/retrieval/search/tests/test_tree_integration.py -v`
Expected: FAIL — `tree_search` parameter not recognized

**Step 3: Modify RetrievalService**

In `src/rfnry_rag/retrieval/modules/retrieval/search/service.py`:

Add `tree_search` parameter to `__init__`:

```python
def __init__(
    self,
    vector_search: VectorSearch,
    keyword_search: KeywordSearch | None = None,
    reranking: BaseReranking | None = None,
    top_k: int = 5,
    source_type_weights: dict[str, float] | None = None,
    document_store: BaseDocumentStore | None = None,
    query_rewriter: BaseQueryRewriter | None = None,
    graph_store: BaseGraphStore | None = None,
    chunk_refiner: BaseChunkRefiner | None = None,
    tree_search: Any | None = None,  # TreeSearchService, typed as Any to avoid circular import
) -> None:
    # ... existing initialization ...
    self._tree_search = tree_search
```

Add tree search to the named tasks in `_search_single_query`:

```python
if self._tree_search:
    named_tasks["tree"] = self._tree_search.search(query)
```

Note: The tree search integration here is simplified. The actual implementation will need to handle loading the `TreeIndex` and `PageContent` for the relevant sources and passing them to the tree search service. This wiring will be completed in Task 13 when integrating with RagServer, which has access to the metadata store and document store needed to load tree indexes and page content.

**Step 4: Run test to verify it passes**

Run: `pytest src/rfnry_rag/retrieval/modules/retrieval/search/tests/test_tree_integration.py -v`
Expected: PASS

**Step 5: Run full test suite to check for regressions**

Run: `pytest src/rfnry_rag/retrieval/tests/ -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/rfnry_rag/retrieval/modules/retrieval/search/service.py src/rfnry_rag/retrieval/modules/retrieval/search/tests/test_tree_integration.py
git commit -m "feat(tree): add tree search path to RetrievalService"
```

---

### Task 13: Integration — RagServer Wiring

Wire tree services into RagServer initialization, ingestion, and retrieval.

**Files:**
- Modify: `src/rfnry_rag/retrieval/server.py`
- Test: `src/rfnry_rag/retrieval/tests/test_tree_server.py`

**Step 1: Write the failing tests**

```python
"""Tests for tree integration in RagServer."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from x64rag.retrieval.server import (
    IngestionConfig,
    PersistenceConfig,
    RagServerConfig,
    TreeIndexingConfig,
    TreeSearchConfig,
)


def _mock_persistence() -> PersistenceConfig:
    return PersistenceConfig(
        vector_store=AsyncMock(),
        metadata_store=AsyncMock(),
    )


def _mock_ingestion() -> IngestionConfig:
    embeddings = MagicMock()
    embeddings.model = "test-embed"
    return IngestionConfig(embeddings=embeddings)


def test_rag_server_config_includes_tree():
    config = RagServerConfig(
        persistence=_mock_persistence(),
        ingestion=_mock_ingestion(),
        tree_indexing=TreeIndexingConfig(enabled=True),
        tree_search=TreeSearchConfig(enabled=True, max_steps=3),
    )
    assert config.tree_indexing.enabled is True
    assert config.tree_search.max_steps == 3


def test_rag_server_config_tree_defaults_disabled():
    config = RagServerConfig(
        persistence=_mock_persistence(),
        ingestion=_mock_ingestion(),
    )
    assert config.tree_indexing.enabled is False
    assert config.tree_search.enabled is False
```

**Step 2: Run test to verify it passes** (config was added in Task 2)

Run: `pytest src/rfnry_rag/retrieval/tests/test_tree_server.py -v`
Expected: PASS

**Step 3: Wire services in RagServer.initialize()**

In `src/rfnry_rag/retrieval/server.py`, add to `initialize()`:

```python
# After existing service initialization...

# Tree indexing service
self._tree_indexing_service: TreeIndexingService | None = None
if self._config.tree_indexing.enabled and self._config.persistence.metadata_store:
    from x64rag.retrieval.modules.ingestion.tree.service import TreeIndexingService
    tree_indexing_registry = build_registry(self._config.tree_indexing.model) if self._config.tree_indexing.model else None
    self._tree_indexing_service = TreeIndexingService(
        config=self._config.tree_indexing,
        metadata_store=self._config.persistence.metadata_store,
        registry=tree_indexing_registry,
    )

# Tree search service
self._tree_search_service: TreeSearchService | None = None
if self._config.tree_search.enabled:
    from x64rag.retrieval.modules.retrieval.tree.service import TreeSearchService
    tree_search_registry = build_registry(self._config.tree_search.model) if self._config.tree_search.model else None
    self._tree_search_service = TreeSearchService(
        config=self._config.tree_search,
        registry=tree_search_registry,
    )
```

**Step 4: Add tree_index parameter to RagServer.ingest()**

```python
async def ingest(
    self,
    file_path: str | Path,
    *,
    knowledge_id: str | None = None,
    source_type: str | None = None,
    metadata: dict[str, Any] | None = None,
    page_range: str | None = None,
    tree_index: bool = False,
    **kwargs,
) -> Source:
    # ... existing ingestion logic ...
    source = await self._unstructured_ingestion.ingest(...)

    # Tree indexing (opt-in)
    if tree_index and self._tree_indexing_service:
        from x64rag.retrieval.modules.ingestion.tree.toc import PageContent
        # Re-parse pages for tree indexing
        parsed_pages = self._parse_pages(file_path, page_range)
        tree_pages = [
            PageContent(index=i, text=page.text, token_count=len(page.text) // 4)
            for i, page in enumerate(parsed_pages)
        ]
        tree = await self._tree_indexing_service.build_tree_index(
            source_id=source.source_id,
            doc_name=Path(file_path).name,
            pages=tree_pages,
        )
        await self._tree_indexing_service.save_tree_index(tree)

    return source
```

**Step 5: Run full test suite**

Run: `pytest src/rfnry_rag/retrieval/ -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/rfnry_rag/retrieval/server.py src/rfnry_rag/retrieval/tests/test_tree_server.py
git commit -m "feat(tree): wire tree services into RagServer"
```

---

### Task 14: Full Integration Test

End-to-end test with mocked LLM calls verifying the complete flow.

**Files:**
- Test: `src/rfnry_rag/retrieval/tests/test_tree_e2e.py`

**Step 1: Write the integration test**

```python
"""End-to-end test for tree indexing and search with mocked BAML calls."""

import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from x64rag.retrieval.common.models import TreeIndex, TreeNode, TreeSearchResult
from x64rag.retrieval.modules.ingestion.tree.service import TreeIndexingService
from x64rag.retrieval.modules.ingestion.tree.toc import PageContent, TocInfo, TocPath
from x64rag.retrieval.modules.retrieval.tree.service import TreeSearchService
from x64rag.retrieval.server import TreeIndexingConfig, TreeSearchConfig


async def test_full_tree_pipeline():
    """Index a document, then search it using tree retrieval."""

    # Setup
    pages = [
        PageContent(index=0, text="Title page of Annual Report 2025", token_count=50),
        PageContent(index=1, text="Table of Contents\n1. Introduction...p3\n2. Financial Results...p5", token_count=100),
        PageContent(index=2, text="Introduction to our company and mission", token_count=200),
        PageContent(index=3, text="More introduction details", token_count=200),
        PageContent(index=4, text="Financial Results for Q1-Q4", token_count=300),
        PageContent(index=5, text="Revenue grew 15% year over year to $2.3B", token_count=300),
        PageContent(index=6, text="Operating expenses and margin analysis", token_count=300),
        PageContent(index=7, text="Conclusion and forward-looking statements", token_count=200),
    ]

    # --- INDEXING ---
    indexing_service = TreeIndexingService(
        config=TreeIndexingConfig(
            enabled=True,
            generate_summaries=False,
            generate_description=False,
        ),
        metadata_store=AsyncMock(),
    )

    toc_info = TocInfo(path=TocPath.WITH_PAGE_NUMBERS, toc_pages=[pages[1]])
    parsed_entries = [
        {"structure": "1", "title": "Introduction", "page": 2},
        {"structure": "2", "title": "Financial Results", "page": 4},
        {"structure": "3", "title": "Conclusion", "page": 7},
    ]
    verified = [{**e, "start_page": e["page"], "verified": True} for e in parsed_entries]

    with (
        patch("x64rag.retrieval.modules.ingestion.tree.service.detect_toc", return_value=toc_info),
        patch("x64rag.retrieval.modules.ingestion.tree.service.parse_toc", return_value=parsed_entries),
        patch("x64rag.retrieval.modules.ingestion.tree.service.verify_section_positions", return_value=verified),
    ):
        tree_index = await indexing_service.build_tree_index(
            source_id="annual-report-2025",
            doc_name="annual_report_2025.pdf",
            pages=pages,
        )

    assert tree_index.source_id == "annual-report-2025"
    assert len(tree_index.structure) == 3
    assert tree_index.structure[0].title == "Introduction"
    assert tree_index.structure[1].title == "Financial Results"

    # --- SEARCH ---
    search_service = TreeSearchService(
        config=TreeSearchConfig(enabled=True, max_steps=3),
    )

    resolved = SimpleNamespace(pages="4-6", reasoning="Financial Results section contains revenue data")

    with patch(
        "x64rag.retrieval.modules.retrieval.tree.service._call_retrieval_step",
        return_value=resolved,
    ):
        results = await search_service.search("What was the revenue?", tree_index, pages)

    assert len(results) == 1
    assert "4-6" in results[0].pages
    assert "revenue" in results[0].content.lower()

    # --- CONVERSION ---
    chunks = search_service.to_retrieved_chunks(results, tree_index)
    assert len(chunks) == 1
    assert chunks[0].source_id == "annual-report-2025"
    assert "revenue" in chunks[0].content.lower()
```

**Step 2: Run the integration test**

Run: `pytest src/rfnry_rag/retrieval/tests/test_tree_e2e.py -v`
Expected: PASS

**Step 3: Run full test suite**

Run: `poe test`
Expected: All tests PASS, including existing 394 tests + new tree tests

**Step 4: Commit**

```bash
git add src/rfnry_rag/retrieval/tests/test_tree_e2e.py
git commit -m "test(tree): add end-to-end integration test for tree pipeline"
```
