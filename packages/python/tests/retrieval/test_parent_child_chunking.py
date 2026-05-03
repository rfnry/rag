from rfnry_knowledge.ingestion.chunk.chunker import SemanticChunker
from rfnry_knowledge.ingestion.models import ParsedPage


def test_parent_child_chunking_produces_both_levels():
    chunker = SemanticChunker(chunk_size=100, chunk_overlap=10, parent_chunk_size=300, parent_chunk_overlap=30)
    pages = [ParsedPage(page_number=1, content="A" * 500)]
    chunks = chunker.chunk(pages)

    parents = [c for c in chunks if c.chunk_type == "parent"]
    children = [c for c in chunks if c.chunk_type == "child"]

    assert len(parents) > 0
    assert len(children) > 0
    for child in children:
        assert child.parent_id is not None


def test_parent_child_disabled():
    chunker = SemanticChunker(chunk_size=100, chunk_overlap=10, parent_chunk_size=0)
    pages = [ParsedPage(page_number=1, content="A" * 500)]
    chunks = chunker.chunk(pages)

    parents = [c for c in chunks if c.chunk_type == "parent"]
    children = [c for c in chunks if c.chunk_type == "child"]

    assert len(parents) == 0
    assert len(children) > 0
    for child in children:
        assert child.parent_id is None


def test_children_share_parent_id():
    chunker = SemanticChunker(chunk_size=50, chunk_overlap=5, parent_chunk_size=200, parent_chunk_overlap=20)
    text = (
        "Sentence one about filters. Sentence two about pressure. "
        "Sentence three about specs. Sentence four about installation. "
        "Sentence five about maintenance."
    )
    pages = [ParsedPage(page_number=1, content=text)]
    chunks = chunker.chunk(pages)

    children = [c for c in chunks if c.chunk_type == "child"]
    parents = [c for c in chunks if c.chunk_type == "parent"]

    parent_ids = {p.parent_id for p in parents}
    for child in children:
        assert child.parent_id in parent_ids


def test_flat_chunking_backward_compat():
    """Default chunker (no parent) works exactly like before."""
    chunker = SemanticChunker(chunk_size=100, chunk_overlap=10)
    pages = [ParsedPage(page_number=1, content="Some content that should be chunked.")]
    chunks = chunker.chunk(pages)

    assert all(c.chunk_type == "child" for c in chunks)
    assert all(c.parent_id is None for c in chunks)
