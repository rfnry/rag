import json
from datetime import datetime

from rfnry_rag.retrieval.common.models import TreeIndex, TreeNode


class TestTreeIndexJsonRoundtrip:
    """Test JSON serialization roundtrip for TreeIndex, validating that
    tree indexes can be serialized to JSON strings and deserialized back
    without data loss — the same path used by the metadata store."""

    def test_simple_roundtrip(self):
        now = datetime(2026, 4, 6, 12, 0, 0)
        node = TreeNode(node_id="n1", title="Chapter 1", start_index=0, end_index=100)
        index = TreeIndex(
            source_id="src-1",
            doc_name="Test Doc",
            doc_description="A test document",
            structure=[node],
            page_count=10,
            created_at=now,
        )

        json_str = json.dumps(index.to_dict())
        restored = TreeIndex.from_dict(json.loads(json_str))

        assert restored.source_id == "src-1"
        assert restored.doc_name == "Test Doc"
        assert restored.doc_description == "A test document"
        assert restored.page_count == 10
        assert restored.created_at == now
        assert len(restored.structure) == 1
        assert restored.structure[0].node_id == "n1"
        assert restored.structure[0].title == "Chapter 1"

    def test_nested_structure_roundtrip(self):
        now = datetime(2026, 1, 15, 8, 30, 0)
        grandchild = TreeNode(node_id="n3", title="Section 1.1.1", start_index=0, end_index=20, summary="Deep section")
        child = TreeNode(
            node_id="n2", title="Section 1.1", start_index=0, end_index=50, summary="Sub section", children=[grandchild]
        )
        root = TreeNode(
            node_id="n1", title="Chapter 1", start_index=0, end_index=100, summary="Top level", children=[child]
        )
        index = TreeIndex(
            source_id="src-nested",
            doc_name="Nested Doc",
            doc_description="Deeply nested structure",
            structure=[root],
            page_count=5,
            created_at=now,
        )

        json_str = json.dumps(index.to_dict())
        restored = TreeIndex.from_dict(json.loads(json_str))

        assert len(restored.structure) == 1
        assert len(restored.structure[0].children) == 1
        assert len(restored.structure[0].children[0].children) == 1
        assert restored.structure[0].children[0].children[0].node_id == "n3"
        assert restored.structure[0].children[0].children[0].summary == "Deep section"

    def test_empty_structure_roundtrip(self):
        now = datetime(2026, 3, 1, 0, 0, 0)
        index = TreeIndex(
            source_id="src-empty",
            doc_name="Empty Doc",
            doc_description=None,
            structure=[],
            page_count=0,
            created_at=now,
        )

        json_str = json.dumps(index.to_dict())
        restored = TreeIndex.from_dict(json.loads(json_str))

        assert restored.source_id == "src-empty"
        assert restored.doc_description is None
        assert restored.structure == []
        assert restored.page_count == 0

    def test_multiple_top_level_nodes_roundtrip(self):
        now = datetime(2026, 6, 15, 14, 0, 0)
        nodes = [
            TreeNode(node_id=f"ch{i}", title=f"Chapter {i}", start_index=i * 100, end_index=(i + 1) * 100 - 1)
            for i in range(5)
        ]
        index = TreeIndex(
            source_id="src-multi",
            doc_name="Multi-Chapter Doc",
            doc_description="Five chapters",
            structure=nodes,
            page_count=50,
            created_at=now,
        )

        json_str = json.dumps(index.to_dict())
        restored = TreeIndex.from_dict(json.loads(json_str))

        assert len(restored.structure) == 5
        for i, node in enumerate(restored.structure):
            assert node.node_id == f"ch{i}"
            assert node.title == f"Chapter {i}"
            assert node.start_index == i * 100
            assert node.end_index == (i + 1) * 100 - 1

    def test_json_string_is_valid_json(self):
        """Verify the serialized output is a valid JSON string (as stored in the DB column)."""
        now = datetime(2026, 4, 6, 12, 0, 0)
        index = TreeIndex(
            source_id="src-json",
            doc_name="JSON Doc",
            doc_description="Verify JSON validity",
            structure=[TreeNode(node_id="n1", title="Root", start_index=0, end_index=10)],
            page_count=1,
            created_at=now,
        )

        json_str = json.dumps(index.to_dict())
        assert isinstance(json_str, str)

        # Should parse without error
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)
        assert parsed["source_id"] == "src-json"
