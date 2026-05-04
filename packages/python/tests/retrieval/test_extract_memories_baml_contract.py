from pathlib import Path

import pytest

BAML_DIR = (
    Path(__file__).resolve().parents[2]
    / "src" / "rfnry_knowledge" / "baml" / "baml_src" / "memory"
)


def test_functions_baml_exists() -> None:
    assert (BAML_DIR / "functions.baml").is_file()


def test_types_baml_exists() -> None:
    assert (BAML_DIR / "types.baml").is_file()


def test_extract_memories_function_declared() -> None:
    src = (BAML_DIR / "functions.baml").read_text()
    assert "function ExtractMemories(" in src
    assert "interaction:" in src
    assert "occurred_at:" in src
    assert "existing_memories:" in src
    assert "ExtractedMemoryList" in src


def test_extract_memories_fences_user_inputs() -> None:
    src = (BAML_DIR / "functions.baml").read_text()
    for name in ("interaction", "occurred_at", "existing_memories"):
        assert "{{ " + name + " }}" in src or "{{" + name + "}}" in src, name
        assert "========" in src


def test_extract_memories_types_declared() -> None:
    src = (BAML_DIR / "types.baml").read_text()
    assert "class ExtractedMemoryFact" in src
    assert "class ExtractedMemoryList" in src
    for field in ("text string", "attributed_to string?", "linked_memory_row_ids string[]"):
        assert field in src
