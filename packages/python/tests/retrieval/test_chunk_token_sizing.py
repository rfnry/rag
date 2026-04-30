"""Token-aware chunk sizing: chunk_size_unit controls whether length is counted in chars or tokens."""

from rfnry_rag.ingestion.chunk.chunker import SemanticChunker
from rfnry_rag.ingestion.models import ParsedPage


def _page(content: str, page_number: int = 1) -> ParsedPage:
    return ParsedPage(page_number=page_number, content=content, metadata={})


def test_token_mode_produces_larger_chars_than_char_mode_for_same_numeric_size() -> None:
    # 2000 chars of English prose. At chunk_size=300, char mode chunks at ~300 chars;
    # token mode at ~300 tokens ≈ 1200 chars per chunk (~4x larger).
    text = "The quick brown fox jumps over the lazy dog. " * 60

    char_chunker = SemanticChunker(chunk_size=300, chunk_size_unit="chars", chunk_overlap=30)
    tok_chunker = SemanticChunker(chunk_size=300, chunk_size_unit="tokens", chunk_overlap=30)

    char_chunks = char_chunker.chunk([_page(text)])
    tok_chunks = tok_chunker.chunk([_page(text)])

    # Token-mode produces fewer but larger chunks
    assert len(tok_chunks) < len(char_chunks)
    avg_char_len = sum(len(c.content) for c in char_chunks) / len(char_chunks)
    avg_tok_len = sum(len(c.content) for c in tok_chunks) / len(tok_chunks)
    assert avg_tok_len > avg_char_len * 2.0


def test_default_chunk_size_unit_is_tokens() -> None:
    # Token mode is the new default; char mode is opt-in.
    chunker = SemanticChunker()
    assert chunker.chunk_size_unit == "tokens"


def test_token_mode_falls_back_to_word_count_when_tiktoken_unavailable(monkeypatch) -> None:
    from rfnry_rag.ingestion.chunk import token_counter

    monkeypatch.setattr(token_counter, "_TIKTOKEN_AVAILABLE", False)
    # word count ≈ tokens/1.3 for English
    assert token_counter.count_tokens("hello world foo") == 3


def test_invalid_chunk_size_unit_raises() -> None:
    import pytest

    with pytest.raises(ValueError, match="chunk_size_unit"):
        SemanticChunker(chunk_size_unit="bogus")  # type: ignore[arg-type]
