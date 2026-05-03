"""Structure-aware chunking: code blocks and tables survive as atomic units; headings populate section."""

from rfnry_knowledge.ingestion.chunk.chunker import SemanticChunker
from rfnry_knowledge.ingestion.models import ParsedPage


def test_code_fence_survives_as_single_chunk_when_under_size() -> None:
    page = ParsedPage(
        page_number=1,
        content=("Preamble text.\n\n```python\ndef hello():\n    return 'world'\n```\n\nTrailing text."),
        metadata={},
    )
    chunker = SemanticChunker(chunk_size=500, chunk_size_unit="chars", chunk_overlap=50)
    chunks = chunker.chunk([page])

    # The code fence is one atomic chunk
    code_chunks = [c for c in chunks if "def hello()" in c.content]
    assert len(code_chunks) == 1
    assert "```python" in code_chunks[0].content
    assert "```" in code_chunks[0].content.split("def hello()")[1]  # closing fence also present


def test_markdown_table_survives_as_atomic_unit() -> None:
    page = ParsedPage(
        page_number=1,
        content=(
            "# Specifications\n\n"
            "| Bolt | Torque | Unit |\n"
            "| --- | --- | --- |\n"
            "| M8 | 24 | N·m |\n"
            "| M10 | 48 | N·m |\n"
            "| M12 | 84 | N·m |\n"
        ),
        metadata={},
    )
    chunker = SemanticChunker(chunk_size=500, chunk_size_unit="chars", chunk_overlap=50)
    chunks = chunker.chunk([page])

    # Table is one chunk that includes both header + all rows
    table_chunks = [c for c in chunks if "Torque" in c.content]
    assert len(table_chunks) == 1
    assert "M8" in table_chunks[0].content and "M12" in table_chunks[0].content


def test_heading_populates_section_field() -> None:
    page = ParsedPage(
        page_number=1,
        content=(
            "# Safety\n\n## Lockout procedures\n\n### Step 2\n\nAlways verify de-energisation before proceeding.\n"
        ),
        metadata={},
    )
    chunker = SemanticChunker(chunk_size=500, chunk_size_unit="chars", chunk_overlap=50)
    chunks = chunker.chunk([page])

    step2_chunk = next(c for c in chunks if "verify de-energisation" in c.content)
    assert step2_chunk.section == "Safety > Lockout procedures > Step 2"


def test_bug1_python_comment_inside_code_fence_not_lifted_as_heading() -> None:
    page = ParsedPage(
        page_number=1,
        content=(
            "# Real heading\n\n"
            "```python\n"
            "# This is a comment, not a heading\n"
            "def foo(): pass\n"
            "```\n\n"
            "Body text after fence.\n"
        ),
        metadata={},
    )
    chunker = SemanticChunker(chunk_size=500, chunk_size_unit="chars")
    chunks = chunker.chunk([page])

    # Body-text chunk's section should be 'Real heading', NOT 'This is a comment'
    body_chunk = next(c for c in chunks if "Body text" in c.content)
    assert body_chunk.section == "Real heading"


def test_bug2_overlap_chunk_still_gets_valid_section() -> None:
    # Construct content where _merge_splits will produce a chunk whose
    # text starts earlier than the current search cursor (due to overlap)
    page = ParsedPage(
        page_number=1,
        content=(
            "# Section\n\n" + ("word " * 200)  # 1000 chars of prose to force splitting
        ),
        metadata={},
    )
    chunker = SemanticChunker(chunk_size=300, chunk_size_unit="chars", chunk_overlap=50)
    chunks = chunker.chunk([page])

    # Every chunk must have the "Section" label — none should be None
    assert all(c.section == "Section" for c in chunks), [c.section for c in chunks]


def test_bug3_atx_closing_hash_stripped_from_section_path() -> None:
    page = ParsedPage(page_number=1, content=("# Safety #\n\nSome body text in Safety section.\n"), metadata={})
    chunker = SemanticChunker(chunk_size=500, chunk_size_unit="chars")
    chunks = chunker.chunk([page])

    body = next(c for c in chunks if "body text" in c.content)
    assert body.section == "Safety"  # NOT "Safety #"


def test_bug4_parent_child_preserves_atomic_tables() -> None:
    page = ParsedPage(
        page_number=1,
        content=(
            "Preamble.\n\n"
            "| a | b |\n"
            "| --- | --- |\n"
            "| 1 | 2 |\n"
            "| 3 | 4 |\n"
            "| 5 | 6 |\n"
            "| 7 | 8 |\n"
            "| 9 | 10 |\n\n"
            "Trailing.\n"
        ),
        metadata={},
    )
    # parent_chunk_size is large enough that the table fits as one parent
    chunker = SemanticChunker(
        chunk_size=100,
        chunk_size_unit="chars",
        parent_chunk_size=500,
        parent_chunk_overlap=50,
    )
    chunks = chunker.chunk([page])
    # Find parent chunks containing table content
    table_parents = [c for c in chunks if c.chunk_type == "parent" and "| 1 | 2 |" in c.content]
    # The table should appear in exactly one parent chunk
    assert len(table_parents) == 1
    # And that parent should also contain the last row
    assert "| 9 | 10 |" in table_parents[0].content
