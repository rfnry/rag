from datetime import datetime

from rfnry_rag.retrieval.common.models import TreeIndex, TreeNode, TreeSearchResult


class TestTreeNode:
    def test_defaults(self):
        node = TreeNode(node_id="n1", title="Chapter 1", start_index=0, end_index=100)
        assert node.node_id == "n1"
        assert node.title == "Chapter 1"
        assert node.start_index == 0
        assert node.end_index == 100
        assert node.summary is None
        assert node.children == []

    def test_with_children(self):
        child1 = TreeNode(node_id="n2", title="Section 1.1", start_index=0, end_index=50)
        child2 = TreeNode(node_id="n3", title="Section 1.2", start_index=51, end_index=100, summary="A summary")
        parent = TreeNode(node_id="n1", title="Chapter 1", start_index=0, end_index=100, children=[child1, child2])
        assert len(parent.children) == 2
        assert parent.children[0].title == "Section 1.1"
        assert parent.children[1].summary == "A summary"

    def test_to_dict_from_dict_roundtrip(self):
        child = TreeNode(node_id="n2", title="Section 1.1", start_index=0, end_index=50, summary="child summary")
        node = TreeNode(
            node_id="n1",
            title="Chapter 1",
            start_index=0,
            end_index=100,
            summary="parent summary",
            children=[child],
        )
        data = node.to_dict()
        restored = TreeNode.from_dict(data)
        assert restored.node_id == node.node_id
        assert restored.title == node.title
        assert restored.start_index == node.start_index
        assert restored.end_index == node.end_index
        assert restored.summary == node.summary
        assert len(restored.children) == 1
        assert restored.children[0].node_id == child.node_id
        assert restored.children[0].summary == child.summary

    def test_from_dict_missing_optional_fields(self):
        data = {"node_id": "n1", "title": "Root", "start_index": 0, "end_index": 10}
        node = TreeNode.from_dict(data)
        assert node.summary is None
        assert node.children == []


class TestTreeIndex:
    def test_to_dict_from_dict_roundtrip(self):
        now = datetime(2026, 4, 6, 12, 0, 0)
        node = TreeNode(node_id="n1", title="Chapter 1", start_index=0, end_index=100)
        index = TreeIndex(
            source_id="src-1",
            doc_name="Test Document",
            doc_description="A test document",
            structure=[node],
            page_count=10,
            created_at=now,
        )
        data = index.to_dict()
        assert data["created_at"] == "2026-04-06T12:00:00"
        assert len(data["structure"]) == 1

        restored = TreeIndex.from_dict(data)
        assert restored.source_id == index.source_id
        assert restored.doc_name == index.doc_name
        assert restored.doc_description == index.doc_description
        assert restored.page_count == index.page_count
        assert restored.created_at == now
        assert len(restored.structure) == 1
        assert restored.structure[0].node_id == "n1"

    def test_none_description(self):
        data = {
            "source_id": "src-2",
            "doc_name": "Doc",
            "structure": [],
            "page_count": 0,
            "created_at": "2026-01-01T00:00:00",
        }
        index = TreeIndex.from_dict(data)
        assert index.doc_description is None
        assert index.structure == []


class TestTreeSearchResult:
    def test_creation(self):
        result = TreeSearchResult(
            node_id="n1",
            title="Chapter 1",
            pages="1-5",
            content="Some content here",
            reasoning="Matched based on semantic similarity",
        )
        assert result.node_id == "n1"
        assert result.title == "Chapter 1"
        assert result.pages == "1-5"
        assert result.content == "Some content here"
        assert result.reasoning == "Matched based on semantic similarity"
