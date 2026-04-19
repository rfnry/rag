"""Tests for TreeSearchService."""

from __future__ import annotations

import types
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

from rfnry_rag.retrieval.common.models import TreeIndex, TreeNode, TreeSearchResult
from rfnry_rag.retrieval.modules.ingestion.tree.toc import PageContent
from rfnry_rag.retrieval.modules.retrieval.tree.service import TreeSearchService


def _make_config(max_steps: int = 5, max_context_tokens: int = 50_000) -> types.SimpleNamespace:
    return types.SimpleNamespace(max_steps=max_steps, max_context_tokens=max_context_tokens)


def _make_tree_index() -> TreeIndex:
    child1 = TreeNode(node_id="1.1", title="Introduction", start_index=1, end_index=3, summary="Intro")
    child2 = TreeNode(node_id="1.2", title="Methods", start_index=4, end_index=7, summary="Methods section")
    root = TreeNode(node_id="1", title="Chapter 1", start_index=1, end_index=10, children=[child1, child2])
    return TreeIndex(
        source_id="src-001",
        doc_name="Test Document",
        doc_description="A test document",
        structure=[root],
        page_count=10,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def _make_pages() -> list[PageContent]:
    return [PageContent(index=i, text=f"Content of page {i}.", token_count=20) for i in range(1, 11)]


class _FakeResolvedPages:
    """Fake class for ToolResolvedPages."""

    def __init__(self, pages: str, reasoning: str) -> None:
        self.pages = pages
        self.reasoning = reasoning


class _FakeFetchPages:
    """Fake class for ToolFetchPages."""

    def __init__(self, pages: str, reasoning: str) -> None:
        self.pages = pages
        self.reasoning = reasoning


class _FakeDrillDown:
    """Fake class for ToolDrillDown."""

    def __init__(self, node_id: str, reasoning: str) -> None:
        self.node_id = node_id
        self.reasoning = reasoning


def _patch_baml(step_returns):
    """Create patches for lazy BAML imports in the service.

    Returns a context manager that patches:
    - rfnry_rag.retrieval.baml.baml_client.async_client (the 'b' source)
    - rfnry_rag.retrieval.baml.baml_client.types (the type classes)

    The service imports these lazily inside search(), so we must patch
    at the source module level.
    """
    mock_b = MagicMock()
    mock_b.TreeRetrievalStep = AsyncMock(side_effect=step_returns)

    class _Ctx:
        def __init__(self):
            self.mock_b = mock_b
            self._patches = []

        def __enter__(self):
            # Patch the BAML async client's 'b' singleton
            p1 = patch("rfnry_rag.retrieval.baml.baml_client.async_client.b", self.mock_b)
            # Patch the types so isinstance checks work
            p2 = patch("rfnry_rag.retrieval.baml.baml_client.types.ToolResolvedPages", _FakeResolvedPages)
            p3 = patch("rfnry_rag.retrieval.baml.baml_client.types.ToolFetchPages", _FakeFetchPages)
            p4 = patch("rfnry_rag.retrieval.baml.baml_client.types.ToolDrillDown", _FakeDrillDown)
            self._patches = [p1, p2, p3, p4]
            for p in self._patches:
                p.start()
            return self

        def __exit__(self, *args):
            for p in self._patches:
                p.stop()

    return _Ctx()


# --- test_search_resolves_in_one_step ---


async def test_search_resolves_in_one_step():
    """LLM returns ToolResolvedPages on the first step."""
    config = _make_config()
    service = TreeSearchService(config, registry=None)
    tree_index = _make_tree_index()
    pages = _make_pages()

    resolved = _FakeResolvedPages(pages="3,4", reasoning="Pages 3-4 contain the answer")

    with _patch_baml([resolved]) as ctx:
        results = await service.search("What are the methods?", tree_index, pages)

    assert len(results) == 1
    assert results[0].pages == "3,4"
    assert results[0].reasoning == "Pages 3-4 contain the answer"
    assert "Content of page 3" in results[0].content
    assert "Content of page 4" in results[0].content
    ctx.mock_b.TreeRetrievalStep.assert_called_once()


# --- test_search_fetch_then_resolve ---


async def test_search_fetch_then_resolve():
    """LLM first fetches pages, then resolves."""
    config = _make_config()
    service = TreeSearchService(config, registry=None)
    tree_index = _make_tree_index()
    pages = _make_pages()

    fetch_result = _FakeFetchPages(pages="1,2", reasoning="Need to see pages 1-2 first")
    resolve_result = _FakeResolvedPages(pages="2", reasoning="Page 2 has the answer")

    with _patch_baml([fetch_result, resolve_result]) as ctx:
        results = await service.search("What is on page 2?", tree_index, pages)

    assert len(results) == 1
    assert results[0].pages == "2"
    assert "Content of page 2" in results[0].content
    assert ctx.mock_b.TreeRetrievalStep.call_count == 2


# --- test_search_max_steps_exceeded ---


async def test_search_max_steps_exceeded():
    """LLM keeps fetching pages and never resolves — hits max_steps."""
    config = _make_config(max_steps=3)
    service = TreeSearchService(config, registry=None)
    tree_index = _make_tree_index()
    pages = _make_pages()

    fetch1 = _FakeFetchPages(pages="1", reasoning="Check page 1")
    fetch2 = _FakeFetchPages(pages="2", reasoning="Check page 2")
    fetch3 = _FakeFetchPages(pages="3", reasoning="Check page 3")

    with _patch_baml([fetch1, fetch2, fetch3]) as ctx:
        results = await service.search("Something obscure", tree_index, pages)

    assert len(results) == 1
    assert "Content of page 1" in results[0].content
    assert "Content of page 3" in results[0].content
    assert "Max steps" in results[0].reasoning
    assert ctx.mock_b.TreeRetrievalStep.call_count == 3


# --- test_search_drill_down ---


async def test_search_drill_down():
    """LLM drills down into a subtree, then resolves."""
    config = _make_config()
    service = TreeSearchService(config, registry=None)
    tree_index = _make_tree_index()
    pages = _make_pages()

    drill_result = _FakeDrillDown(node_id="1", reasoning="Need to look inside Chapter 1")
    resolve_result = _FakeResolvedPages(pages="4,5", reasoning="Found in Methods section")

    with _patch_baml([drill_result, resolve_result]) as ctx:
        results = await service.search("What methods are used?", tree_index, pages)

    assert len(results) == 1
    assert results[0].pages == "4,5"
    assert "Content of page 4" in results[0].content
    assert ctx.mock_b.TreeRetrievalStep.call_count == 2


# --- test_to_retrieved_chunks ---


def test_to_retrieved_chunks():
    """Convert TreeSearchResult list to RetrievedChunk list."""
    tree_index = _make_tree_index()
    results = [
        TreeSearchResult(
            node_id="1.1",
            title="Introduction",
            pages="1-3",
            content="Some intro content",
            reasoning="Relevant to query",
        ),
        TreeSearchResult(
            node_id="1.2",
            title="Methods",
            pages="4-7",
            content="Methods content",
            reasoning="Also relevant",
        ),
    ]

    chunks = TreeSearchService.to_retrieved_chunks(results, tree_index)

    assert len(chunks) == 2

    assert chunks[0].chunk_id == "tree-1.1-0"
    assert chunks[0].source_id == "src-001"
    assert chunks[0].content == "Some intro content"
    assert chunks[0].score == 1.0
    assert chunks[0].source_metadata["name"] == "Test Document"
    assert chunks[0].source_metadata["tree_pages"] == "1-3"
    assert chunks[0].source_metadata["tree_reasoning"] == "Relevant to query"

    assert chunks[1].chunk_id == "tree-1.2-1"
    assert chunks[1].source_id == "src-001"
    assert chunks[1].content == "Methods content"
    assert chunks[1].score == 1.0
    assert chunks[1].source_metadata["name"] == "Test Document"
    assert chunks[1].source_metadata["tree_pages"] == "4-7"
    assert chunks[1].source_metadata["tree_reasoning"] == "Also relevant"
