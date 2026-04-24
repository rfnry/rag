from pathlib import Path


def test_answer_baml_source_has_content_boundary() -> None:
    """Ingested document content must be fenced in the GenerateAnswer prompt."""
    baml_src = Path("src/rfnry_rag/retrieval/baml/baml_src/generation/answer_functions.baml")
    content = baml_src.read_text()
    assert "GenerateAnswer" in content
    assert "======== CONTEXT START ========" in content
    assert "======== CONTEXT END ========" in content
    # Context fence must appear BEFORE the Question: line
    start = content.index("======== CONTEXT END ========")
    qidx = content.index("Question:", start)
    assert qidx > start


def test_rewriting_baml_source_fences_untrusted_query() -> None:
    """HyDE / multi-query / step-back prompts must fence {{ query }}."""
    src = Path("src/rfnry_rag/retrieval/baml/baml_src/retrieval/rewriting_functions.baml").read_text()
    for fn in ("GenerateHypotheticalDocument", "GenerateQueryVariants", "GenerateStepBackQuery"):
        assert fn in src, f"missing function {fn}"
    # Fences must appear at least three times (once per function).
    assert src.count("======== QUERY START ========") >= 3
    assert src.count("======== QUERY END ========") >= 3
