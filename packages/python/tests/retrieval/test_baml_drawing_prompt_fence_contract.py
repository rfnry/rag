"""BAML drawing functions: prompt-fencing contract + shape."""

from pathlib import Path

_BAML_SRC = Path("src/rfnry_rag/retrieval/baml/baml_src/ingestion/drawing.baml")


def test_drawing_baml_file_exists() -> None:
    assert _BAML_SRC.exists(), f"Missing: {_BAML_SRC}"


def test_drawing_page_analysis_class_declares_required_fields() -> None:
    text = _BAML_SRC.read_text()
    # Required field names must appear in class DrawingPageAnalysis
    for field in [
        "page_number",
        "sheet_number",
        "zone_grid",
        "domain",
        "components",
        "connections",
        "off_page_connectors",
        "title_block",
        "notes",
        "page_type",
    ]:
        assert field in text, f"DrawingPageAnalysis missing field: {field}"


def test_symbol_library_parameter_fenced() -> None:
    """The consumer-supplied symbol_library parameter must be fenced."""
    text = _BAML_SRC.read_text()
    assert "======== SYMBOL LIBRARY START ========" in text
    assert "======== SYMBOL LIBRARY END ========" in text


def test_off_page_patterns_parameter_fenced() -> None:
    text = _BAML_SRC.read_text()
    assert "======== OFF PAGE CONNECTOR PATTERNS START ========" in text
    assert "======== OFF PAGE CONNECTOR PATTERNS END ========" in text
