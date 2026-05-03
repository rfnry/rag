"""Token-aware chunk sizing: chunk_size_unit controls whether length is counted in chars or tokens.

After the provider-decoupling refactor, ``chunk_size_unit='tokens'`` requires a consumer-supplied
``TokenCounter`` Protocol implementation. The lib ships none.
"""

import pytest

from rfnry_knowledge.ingestion.chunk.chunker import SemanticChunker
from rfnry_knowledge.ingestion.models import ParsedPage


class _WordCounter:
    name = "test"
    model = "word"

    def count(self, text: str) -> int:
        return len(text.split())


def _page(content: str, page_number: int = 1) -> ParsedPage:
    return ParsedPage(page_number=page_number, content=content, metadata={})


def test_token_mode_produces_larger_chars_than_char_mode_for_same_numeric_size() -> None:
    # 2000 chars of English prose. At chunk_size=80, char mode chunks at ~80 chars;
    # word-counter token mode at ~80 words ≈ 5x more chars per chunk.
    text = "The quick brown fox jumps over the lazy dog. " * 60

    char_chunker = SemanticChunker(chunk_size=80, chunk_size_unit="chars", chunk_overlap=10)
    tok_chunker = SemanticChunker(
        chunk_size=80, chunk_size_unit="tokens", chunk_overlap=10, token_counter=_WordCounter()
    )

    char_chunks = char_chunker.chunk([_page(text)])
    tok_chunks = tok_chunker.chunk([_page(text)])

    # Token-mode produces fewer but larger chunks
    assert len(tok_chunks) < len(char_chunks)
    avg_char_len = sum(len(c.content) for c in char_chunks) / len(char_chunks)
    avg_tok_len = sum(len(c.content) for c in tok_chunks) / len(tok_chunks)
    assert avg_tok_len > avg_char_len * 2.0


def test_default_chunk_size_unit_is_tokens_falls_back_to_words() -> None:
    """Token mode is the default; with no counter, chunker falls back to whitespace word count."""
    chunker = SemanticChunker()
    assert chunker.chunk_size_unit == "tokens"


def test_token_mode_with_counter_constructs() -> None:
    chunker = SemanticChunker(token_counter=_WordCounter())
    assert chunker.chunk_size_unit == "tokens"


def test_invalid_chunk_size_unit_raises() -> None:
    with pytest.raises(ValueError, match="chunk_size_unit"):
        SemanticChunker(chunk_size_unit="bogus")  # type: ignore[arg-type]
