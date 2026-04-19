"""Tool execution handlers for tree search."""

from __future__ import annotations

from rfnry_rag.retrieval.common.models import TreeNode
from rfnry_rag.retrieval.modules.ingestion.tree.toc import PageContent


def parse_page_ranges(pages_str: str) -> list[int]:
    """Parse a page range string into a sorted list of integers.

    Args:
        pages_str: A string like "3,5-7,12" or "5" or "5-7".
            Whitespace around numbers, commas, and dashes is tolerated.

    Returns:
        Sorted list of unique page indices.
    """
    result: set[int] = set()
    for part in pages_str.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            start = int(start_str.strip())
            end = int(end_str.strip())
            if end >= start:
                result.update(range(start, end + 1))
        else:
            result.add(int(part))
    return sorted(result)


def fetch_pages(pages_str: str, pages: list[PageContent]) -> str:
    """Fetch and format pages by index from a page range string.

    Args:
        pages_str: A page range string (e.g. "3,5-7,12").
        pages: List of PageContent objects to look up by index.

    Returns:
        Formatted string with page headers and text content.
    """
    indices = parse_page_ranges(pages_str)
    pages_by_index = {p.index: p for p in pages}
    sections: list[str] = []
    for idx in indices:
        page = pages_by_index.get(idx)
        if page is not None:
            sections.append(f"--- Page {idx} ---\n{page.text}")
    return "\n\n".join(sections)


def get_subtree(nodes: list[TreeNode], node_id: str) -> TreeNode | None:
    """Recursively search for a node by node_id in a tree.

    Args:
        nodes: List of root-level TreeNode objects.
        node_id: The node_id to search for.

    Returns:
        The matching TreeNode, or None if not found.
    """
    for node in nodes:
        if node.node_id == node_id:
            return node
        found = get_subtree(node.children, node_id)
        if found is not None:
            return found
    return None


def serialize_tree_for_prompt(nodes: list[TreeNode], indent: int = 0) -> str:
    """Convert a tree of nodes to compact text for LLM prompts.

    Format per node:
        [node_id] Title (pages start-end)
          Summary: ...            # only if summary is not None
          <children, indented>

    Uses 2-space indentation per level.

    Args:
        nodes: List of TreeNode objects at the current level.
        indent: Current indentation level (number of 2-space units).

    Returns:
        Multi-line string representation of the tree.
    """
    lines: list[str] = []
    prefix = "  " * indent
    for node in nodes:
        lines.append(f"{prefix}[{node.node_id}] {node.title} (pages {node.start_index}-{node.end_index})")
        if node.summary is not None:
            lines.append(f"{prefix}  Summary: {node.summary}")
        if node.children:
            lines.append(serialize_tree_for_prompt(node.children, indent + 1))
    return "\n".join(lines)
