"""Tree structure building, page range calculation, and large node splitting."""

from __future__ import annotations

from typing import Any

from rfnry_rag.retrieval.common.models import TreeNode


def build_tree(sections: list[dict[str, Any]]) -> list[TreeNode]:
    """Build a tree of TreeNodes from a flat list of section dicts.

    Each section dict must have "structure" (dot-notation like "1", "1.1", "1.1.2"),
    "title", and "page" keys. The dot notation determines parent-child relationships:
    - "1" is a root node
    - "1.1" is a child of "1"
    - "1.1.2" is a child of "1.1"

    Returns only root-level nodes with children nested inside them.
    """
    # Map from structure string to TreeNode for fast parent lookup
    node_map: dict[str, TreeNode] = {}
    roots: list[TreeNode] = []
    for counter, section in enumerate(sections, start=1):
        structure: str = section["structure"]
        title: str = section["title"]
        page: int = section["page"]

        node_id = f"{counter:04d}"

        node = TreeNode(
            node_id=node_id,
            title=title,
            start_index=page,
            end_index=0,
        )
        node_map[structure] = node

        # Determine parent by stripping the last dot segment
        parts = structure.rsplit(".", 1)
        if len(parts) == 1:
            # No dot — this is a root node
            roots.append(node)
        else:
            parent_structure = parts[0]
            parent = node_map.get(parent_structure)
            if parent is not None:
                parent.children.append(node)
            else:
                # Orphan — no matching parent found, treat as root
                roots.append(node)

    return roots


def calculate_page_ranges(nodes: list[TreeNode], total_pages: int) -> None:
    """Set end_index for each node based on sibling boundaries and total page count.

    Mutates nodes in place. For each node at a given level:
    - If there is a next sibling, end_index = next_sibling.start_index - 1
    - If it is the last node at its level, end_index = total_pages

    Recurses into children.
    """
    for i, node in enumerate(nodes):
        if i + 1 < len(nodes):
            node.end_index = nodes[i + 1].start_index - 1
        else:
            node.end_index = total_pages

        if node.children:
            calculate_page_ranges(node.children, node.end_index)


def split_large_nodes(
    nodes: list[TreeNode],
    pages: list[Any],
    max_pages: int,
    max_tokens: int,
) -> list[TreeNode]:
    """Split leaf nodes that exceed both max_pages AND max_tokens.

    For qualifying leaf nodes, splits at the midpoint into two child nodes
    ("Part 1" and "Part 2"), then recurses to keep splitting if still too large.

    Non-leaf nodes recurse into their children without splitting.

    Args:
        nodes: List of TreeNodes to process.
        pages: List of page-like objects with .index and .token_count attributes.
        max_pages: Maximum number of pages before a node is a split candidate.
        max_tokens: Maximum total tokens before a node is a split candidate.

    Returns:
        The processed list of nodes (same list, mutated in place, but also returned).
    """
    result: list[TreeNode] = []

    for node in nodes:
        if node.children:
            # Non-leaf: recurse into children
            node.children = split_large_nodes(node.children, pages, max_pages, max_tokens)
            result.append(node)
            continue

        # Leaf node — check if it exceeds both limits
        node_pages = [p for p in pages if p.index >= node.start_index and p.index <= node.end_index]
        page_count = len(node_pages)
        token_count = sum(p.token_count for p in node_pages)

        if page_count > max_pages and token_count > max_tokens:
            # Split at midpoint
            mid = node.start_index + (node.end_index - node.start_index) // 2

            child1 = TreeNode(
                node_id=f"{node.node_id}-1",
                title=f"{node.title} — Part 1",
                start_index=node.start_index,
                end_index=mid,
            )
            child2 = TreeNode(
                node_id=f"{node.node_id}-2",
                title=f"{node.title} — Part 2",
                start_index=mid + 1,
                end_index=node.end_index,
            )

            node.children = split_large_nodes([child1, child2], pages, max_pages, max_tokens)
            result.append(node)
        else:
            result.append(node)

    return result
