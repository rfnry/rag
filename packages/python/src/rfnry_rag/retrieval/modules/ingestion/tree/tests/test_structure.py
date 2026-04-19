"""Tests for tree structure building, page range calculation, and large node splitting."""

from dataclasses import dataclass

from rfnry_rag.retrieval.modules.ingestion.tree.structure import (
    build_tree,
    calculate_page_ranges,
    split_large_nodes,
)


@dataclass
class MockPage:
    """Simple mock for page objects with index and token_count."""

    index: int
    token_count: int


class TestBuildTreeFlatSections:
    def test_build_tree_flat_sections(self):
        """Flat sections (no nesting) should all be root nodes."""
        sections = [
            {"structure": "1", "title": "Chapter 1", "page": 1},
            {"structure": "2", "title": "Chapter 2", "page": 10},
            {"structure": "3", "title": "Chapter 3", "page": 20},
        ]
        roots = build_tree(sections)
        assert len(roots) == 3
        assert roots[0].title == "Chapter 1"
        assert roots[0].node_id == "0001"
        assert roots[0].start_index == 1
        assert roots[1].title == "Chapter 2"
        assert roots[1].node_id == "0002"
        assert roots[1].start_index == 10
        assert roots[2].title == "Chapter 3"
        assert roots[2].node_id == "0003"
        assert roots[2].start_index == 20
        for root in roots:
            assert root.children == []


class TestBuildTreeNestedSections:
    def test_build_tree_nested_sections(self):
        """Parent-child relationships via dot notation."""
        sections = [
            {"structure": "1", "title": "Chapter 1", "page": 1},
            {"structure": "1.1", "title": "Section 1.1", "page": 3},
            {"structure": "1.2", "title": "Section 1.2", "page": 7},
            {"structure": "2", "title": "Chapter 2", "page": 15},
        ]
        roots = build_tree(sections)
        assert len(roots) == 2
        assert roots[0].title == "Chapter 1"
        assert len(roots[0].children) == 2
        assert roots[0].children[0].title == "Section 1.1"
        assert roots[0].children[0].node_id == "0002"
        assert roots[0].children[1].title == "Section 1.2"
        assert roots[0].children[1].node_id == "0003"
        assert roots[1].title == "Chapter 2"
        assert roots[1].children == []


class TestBuildTreeDeepNesting:
    def test_build_tree_deep_nesting(self):
        """3-level nesting: root -> child -> grandchild."""
        sections = [
            {"structure": "1", "title": "Part 1", "page": 1},
            {"structure": "1.1", "title": "Chapter 1.1", "page": 5},
            {"structure": "1.1.1", "title": "Section 1.1.1", "page": 6},
            {"structure": "1.1.2", "title": "Section 1.1.2", "page": 10},
            {"structure": "1.2", "title": "Chapter 1.2", "page": 15},
        ]
        roots = build_tree(sections)
        assert len(roots) == 1
        root = roots[0]
        assert root.title == "Part 1"
        assert len(root.children) == 2

        ch1 = root.children[0]
        assert ch1.title == "Chapter 1.1"
        assert len(ch1.children) == 2
        assert ch1.children[0].title == "Section 1.1.1"
        assert ch1.children[0].node_id == "0003"
        assert ch1.children[1].title == "Section 1.1.2"
        assert ch1.children[1].node_id == "0004"

        ch2 = root.children[1]
        assert ch2.title == "Chapter 1.2"
        assert ch2.children == []


class TestCalculatePageRanges:
    def test_calculate_page_ranges(self):
        """Basic case: siblings get end_index from next sibling, last gets total_pages."""
        sections = [
            {"structure": "1", "title": "Chapter 1", "page": 1},
            {"structure": "2", "title": "Chapter 2", "page": 10},
            {"structure": "3", "title": "Chapter 3", "page": 20},
        ]
        roots = build_tree(sections)
        calculate_page_ranges(roots, total_pages=30)

        assert roots[0].start_index == 1
        assert roots[0].end_index == 9
        assert roots[1].start_index == 10
        assert roots[1].end_index == 19
        assert roots[2].start_index == 20
        assert roots[2].end_index == 30

    def test_calculate_page_ranges_multiple_roots(self):
        """Multiple roots with children get correct ranges at both levels."""
        sections = [
            {"structure": "1", "title": "Part A", "page": 1},
            {"structure": "1.1", "title": "Section A.1", "page": 1},
            {"structure": "1.2", "title": "Section A.2", "page": 5},
            {"structure": "2", "title": "Part B", "page": 10},
        ]
        roots = build_tree(sections)
        calculate_page_ranges(roots, total_pages=20)

        # Root level
        assert roots[0].end_index == 9
        assert roots[1].end_index == 20

        # Children of Part A — bounded by parent's end_index
        children = roots[0].children
        assert len(children) == 2
        assert children[0].start_index == 1
        assert children[0].end_index == 4
        assert children[1].start_index == 5
        assert children[1].end_index == 9


class TestSplitLargeNodes:
    def test_split_large_nodes_no_split_needed(self):
        """Nodes within limits should not be split."""
        sections = [
            {"structure": "1", "title": "Chapter 1", "page": 1},
            {"structure": "2", "title": "Chapter 2", "page": 5},
        ]
        roots = build_tree(sections)
        calculate_page_ranges(roots, total_pages=8)

        # 8 pages, each with 100 tokens
        pages = [MockPage(index=i, token_count=100) for i in range(1, 9)]

        result = split_large_nodes(roots, pages, max_pages=10, max_tokens=5000)

        assert len(result) == 2
        assert result[0].children == []
        assert result[1].children == []
        assert result[0].title == "Chapter 1"
        assert result[1].title == "Chapter 2"

    def test_split_large_nodes_splits_when_both_exceeded(self):
        """A leaf node exceeding both max_pages and max_tokens gets split."""
        sections = [
            {"structure": "1", "title": "Big Chapter", "page": 1},
        ]
        roots = build_tree(sections)
        calculate_page_ranges(roots, total_pages=20)

        # 20 pages, each with 1000 tokens = 20,000 total tokens
        pages = [MockPage(index=i, token_count=1000) for i in range(1, 21)]

        result = split_large_nodes(roots, pages, max_pages=5, max_tokens=3000)

        # The root should now have children from splitting
        assert len(result) == 1
        assert len(result[0].children) > 0
        # Verify part titles
        assert "Part 1" in result[0].children[0].title
        assert "Part 2" in result[0].children[1].title

    def test_split_large_nodes_no_split_when_only_pages_exceeded(self):
        """A leaf node exceeding only max_pages (but not max_tokens) should NOT be split."""
        sections = [
            {"structure": "1", "title": "Chapter 1", "page": 1},
        ]
        roots = build_tree(sections)
        calculate_page_ranges(roots, total_pages=20)

        # 20 pages, each with only 10 tokens = 200 total tokens
        pages = [MockPage(index=i, token_count=10) for i in range(1, 21)]

        result = split_large_nodes(roots, pages, max_pages=5, max_tokens=5000)

        assert len(result) == 1
        assert result[0].children == []

    def test_split_large_nodes_no_split_when_only_tokens_exceeded(self):
        """A leaf node exceeding only max_tokens (but not max_pages) should NOT be split."""
        sections = [
            {"structure": "1", "title": "Chapter 1", "page": 1},
        ]
        roots = build_tree(sections)
        calculate_page_ranges(roots, total_pages=3)

        # 3 pages, each with 5000 tokens = 15,000 total tokens
        pages = [MockPage(index=i, token_count=5000) for i in range(1, 4)]

        result = split_large_nodes(roots, pages, max_pages=10, max_tokens=3000)

        assert len(result) == 1
        assert result[0].children == []

    def test_split_large_nodes_recurses_into_non_leaf(self):
        """Non-leaf nodes should recurse into children without splitting the parent."""
        sections = [
            {"structure": "1", "title": "Part 1", "page": 1},
            {"structure": "1.1", "title": "Big Section", "page": 1},
        ]
        roots = build_tree(sections)
        calculate_page_ranges(roots, total_pages=20)

        # 20 pages, each with 1000 tokens
        pages = [MockPage(index=i, token_count=1000) for i in range(1, 21)]

        result = split_large_nodes(roots, pages, max_pages=5, max_tokens=3000)

        # Parent stays as-is, but its child (leaf) should have been split
        assert len(result) == 1
        assert result[0].title == "Part 1"
        child = result[0].children[0]
        assert child.title == "Big Section"
        assert len(child.children) > 0
