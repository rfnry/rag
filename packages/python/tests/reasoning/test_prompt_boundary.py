from pathlib import Path


def test_reasoning_analyze_context_baml_fences_roles() -> None:
    src = Path("src/rfnry_rag/reasoning/baml/baml_src/analysis/functions.baml").read_text()
    assert "AnalyzeContext" in src
    assert "======== ROLES START ========" in src
    assert "======== ROLES END ========" in src


def test_reasoning_check_compliance_baml_fences_dimensions() -> None:
    src = Path("src/rfnry_rag/reasoning/baml/baml_src/compliance/functions.baml").read_text()
    assert "CheckCompliance" in src
    assert "======== DIMENSIONS START ========" in src
    assert "======== DIMENSIONS END ========" in src
