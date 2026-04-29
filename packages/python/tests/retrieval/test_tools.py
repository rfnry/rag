"""Tests for tree search tool handlers."""

from __future__ import annotations

from rfnry_rag.retrieval.common.models import TreeNode
from rfnry_rag.retrieval.modules.ingestion.tree.toc import PageContent
from rfnry_rag.retrieval.modules.retrieval.tree.tools import (
    fetch_pages,
    get_subtree,
    parse_page_ranges,
    serialize_tree_for_prompt,
)

# --- parse_page_ranges ---


def test_parse_page_ranges_single():
    assert parse_page_ranges("5") == [5]


def test_parse_page_ranges_range():
    assert parse_page_ranges("5-7") == [5, 6, 7]


def test_parse_page_ranges_mixed():
    assert parse_page_ranges("3,5-7,12") == [3, 5, 6, 7, 12]


def test_parse_page_ranges_whitespace():
    assert parse_page_ranges(" 3 , 5 - 7 ") == [3, 5, 6, 7]


# --- fetch_pages ---


def test_fetch_pages():
    pages = [
        PageContent(index=1, text="Page one text", token_count=10),
        PageContent(index=2, text="Page two text", token_count=10),
        PageContent(index=3, text="Page three text", token_count=10),
    ]
    result = fetch_pages("1,3", pages)
    assert result == "--- Page 1 ---\nPage one text\n\n--- Page 3 ---\nPage three text"


# --- get_subtree ---


def _make_tree() -> list[TreeNode]:
    child = TreeNode(node_id="1.1", title="Child", start_index=1, end_index=3)
    grandchild = TreeNode(node_id="1.1.1", title="Grandchild", start_index=1, end_index=2)
    child.children = [grandchild]
    root = TreeNode(node_id="1", title="Root", start_index=1, end_index=10, children=[child])
    return [root]


def test_get_subtree():
    nodes = _make_tree()
    found = get_subtree(nodes, "1.1.1")
    assert found is not None
    assert found.title == "Grandchild"


def test_get_subtree_not_found():
    nodes = _make_tree()
    assert get_subtree(nodes, "99") is None


# --- serialize_tree_for_prompt ---


def test_serialize_tree_for_prompt():
    child = TreeNode(
        node_id="1.1",
        title="Introduction",
        start_index=1,
        end_index=3,
        summary="An intro section",
    )
    root = TreeNode(
        node_id="1",
        title="Chapter 1",
        start_index=1,
        end_index=10,
        summary="First chapter",
        children=[child],
    )
    result = serialize_tree_for_prompt([root])
    lines = result.split("\n")
    assert lines[0] == "[1] Chapter 1 (pages 1-10)"
    assert lines[1] == "  Summary: First chapter"
    assert lines[2] == "  [1.1] Introduction (pages 1-3)"
    assert lines[3] == "    Summary: An intro section"


def test_serialize_tree_for_prompt_no_summary():
    node = TreeNode(node_id="1", title="Chapter 1", start_index=1, end_index=5)
    result = serialize_tree_for_prompt([node])
    assert result == "[1] Chapter 1 (pages 1-5)"
    assert "Summary" not in result
