from pathlib import Path


def test_answer_baml_source_has_content_boundary() -> None:
    """Ingested document content and query must both be fenced in GenerateAnswer."""
    baml_src = Path("src/rfnry_rag/baml/baml_src/generation/answer_functions.baml")
    content = baml_src.read_text()
    assert "GenerateAnswer" in content
    # Both query and context must be fenced.
    assert "======== QUERY START ========" in content
    assert "======== QUERY END ========" in content
    assert "======== CONTEXT START ========" in content
    assert "======== CONTEXT END ========" in content
    # Query fence must appear BEFORE the context fence.
    query_end = content.index("======== QUERY END ========")
    context_start = content.index("======== CONTEXT START ========")
    assert context_start > query_end
