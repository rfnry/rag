"""Hard-split guard: no chunk exceeds chunk_size * 1.2 even with no separators in text."""

from rfnry_rag.retrieval.modules.ingestion.chunk.splitter import RecursiveTextSplitter


def test_long_token_with_no_separators_is_hard_split() -> None:
    # 10 000-char base64-like blob with zero separators
    blob = "A" * 10_000
    splitter = RecursiveTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(blob)

    # Every chunk must be <= chunk_size * 1.2 (20 % slack for overlap math)
    assert all(len(c) <= 500 * 1.2 for c in chunks), [len(c) for c in chunks]


def test_hard_split_metadata_flagged_on_chunked_content() -> None:
    from rfnry_rag.retrieval.modules.ingestion.chunk.chunker import SemanticChunker
    from rfnry_rag.retrieval.modules.ingestion.models import ParsedPage

    # Long blob → will require hard-splitting
    blob = "A" * 4_000
    chunker = SemanticChunker(chunk_size=500, chunk_size_unit="chars", chunk_overlap=50)
    chunks = chunker.chunk([ParsedPage(page_number=1, content=blob, metadata={})])

    # Every chunk must carry the char-level hard-split flag
    assert all(c.was_hard_split for c in chunks), [c.was_hard_split for c in chunks]


def test_normal_prose_does_not_trigger_hard_split() -> None:
    from rfnry_rag.retrieval.modules.ingestion.chunk.chunker import SemanticChunker
    from rfnry_rag.retrieval.modules.ingestion.models import ParsedPage

    prose = "This is a sentence. " * 50
    chunker = SemanticChunker(chunk_size=500, chunk_size_unit="chars", chunk_overlap=50)
    chunks = chunker.chunk([ParsedPage(page_number=1, content=prose, metadata={})])

    assert not any(c.was_hard_split for c in chunks)
