import pytest


def test_tree_index_from_dict_rejects_excessive_depth() -> None:
    from rfnry_rag.retrieval.common.models import TreeIndex

    # Build a deeply-nested dict (200 levels) to trigger the depth guard.
    deep: dict = {"node_id": "leaf", "title": "leaf", "start_index": 0, "end_index": 0, "children": []}
    for _ in range(200):
        deep = {"node_id": "n", "title": "n", "start_index": 0, "end_index": 0, "children": [deep]}

    tampered = {
        "source_id": "s1",
        "doc_name": "x",
        "doc_description": "",
        "structure": [deep],
        "pages": [],
        "page_count": 0,
        "created_at": "2026-01-01T00:00:00",
    }
    with pytest.raises(ValueError, match="tree index depth"):
        TreeIndex.from_dict(tampered)


def test_tree_index_from_dict_rejects_excessive_node_count() -> None:
    """Build a wide (not deep) tree with 12 000 flat children — exceeds _MAX_TREE_NODES (10 000)."""
    from rfnry_rag.retrieval.common.models import TreeIndex

    leaf = {"node_id": "l", "title": "l", "start_index": 0, "end_index": 0, "children": []}
    tampered = {
        "source_id": "s2",
        "doc_name": "y",
        "doc_description": "",
        "structure": [
            {
                "node_id": "root",
                "title": "root",
                "start_index": 0,
                "end_index": 0,
                "children": [dict(leaf) for _ in range(12_000)],
            }
        ],
        "pages": [],
        "page_count": 0,
        "created_at": "2026-01-01T00:00:00",
    }
    with pytest.raises(ValueError, match="tree index node count"):
        TreeIndex.from_dict(tampered)
