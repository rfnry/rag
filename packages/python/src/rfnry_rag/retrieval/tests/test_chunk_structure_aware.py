"""Structure-aware chunking: code blocks and tables survive as atomic units; headings populate section."""
from rfnry_rag.retrieval.modules.ingestion.chunk.chunker import SemanticChunker
from rfnry_rag.retrieval.modules.ingestion.models import ParsedPage


def test_code_fence_survives_as_single_chunk_when_under_size() -> None:
    page = ParsedPage(page_number=1, content=(
        "Preamble text.\n\n"
        "```python\n"
        "def hello():\n"
        "    return 'world'\n"
        "```\n\n"
        "Trailing text."
    ), metadata={})
    chunker = SemanticChunker(chunk_size=500, chunk_size_unit="chars", chunk_overlap=50)
    chunks = chunker.chunk([page])

    # The code fence is one atomic chunk
    code_chunks = [c for c in chunks if "def hello()" in c.content]
    assert len(code_chunks) == 1
    assert "```python" in code_chunks[0].content
    assert "```" in code_chunks[0].content.split("def hello()")[1]  # closing fence also present


def test_markdown_table_survives_as_atomic_unit() -> None:
    page = ParsedPage(page_number=1, content=(
        "# Specifications\n\n"
        "| Bolt | Torque | Unit |\n"
        "| --- | --- | --- |\n"
        "| M8 | 24 | N·m |\n"
        "| M10 | 48 | N·m |\n"
        "| M12 | 84 | N·m |\n"
    ), metadata={})
    chunker = SemanticChunker(chunk_size=500, chunk_size_unit="chars", chunk_overlap=50)
    chunks = chunker.chunk([page])

    # Table is one chunk that includes both header + all rows
    table_chunks = [c for c in chunks if "Torque" in c.content]
    assert len(table_chunks) == 1
    assert "M8" in table_chunks[0].content and "M12" in table_chunks[0].content


def test_heading_populates_section_field() -> None:
    page = ParsedPage(page_number=1, content=(
        "# Safety\n\n"
        "## Lockout procedures\n\n"
        "### Step 2\n\n"
        "Always verify de-energisation before proceeding.\n"
    ), metadata={})
    chunker = SemanticChunker(chunk_size=500, chunk_size_unit="chars", chunk_overlap=50)
    chunks = chunker.chunk([page])

    step2_chunk = next(c for c in chunks if "verify de-energisation" in c.content)
    assert step2_chunk.section == "Safety > Lockout procedures > Step 2"
