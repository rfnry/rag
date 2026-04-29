from rfnry_rag.retrieval.modules.ingestion.chunk.splitter import RecursiveTextSplitter


class TestBasicSplitting:
    def test_short_text_returns_single_chunk(self):
        splitter = RecursiveTextSplitter(chunk_size=100, chunk_overlap=0)
        assert splitter.split_text("Hello world") == ["Hello world"]

    def test_splits_on_double_newline(self):
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        splitter = RecursiveTextSplitter(chunk_size=30, chunk_overlap=0)
        chunks = splitter.split_text(text)
        assert len(chunks) >= 2
        assert "Paragraph one." in chunks[0]

    def test_falls_back_to_single_newline(self):
        text = "Line one.\nLine two.\nLine three.\nLine four."
        splitter = RecursiveTextSplitter(chunk_size=25, chunk_overlap=0)
        chunks = splitter.split_text(text)
        assert len(chunks) >= 2

    def test_empty_text(self):
        splitter = RecursiveTextSplitter(chunk_size=100, chunk_overlap=0)
        assert splitter.split_text("") == []

    def test_whitespace_only_text(self):
        splitter = RecursiveTextSplitter(chunk_size=100, chunk_overlap=0)
        assert splitter.split_text("   \n\n   ") == []

    def test_character_fallback(self):
        text = "abcdefghijklmnopqrstuvwxyz"
        splitter = RecursiveTextSplitter(chunk_size=10, chunk_overlap=0)
        chunks = splitter.split_text(text)
        assert all(len(c) <= 10 for c in chunks)
        assert "".join(chunks) == text

    def test_sentence_separator(self):
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        splitter = RecursiveTextSplitter(chunk_size=40, chunk_overlap=0)
        chunks = splitter.split_text(text)
        assert len(chunks) >= 2
        assert "First sentence." in chunks[0]


class TestOverlap:
    def test_overlap_preserves_context(self):
        text = "A.\n\nB.\n\nC.\n\nD."
        splitter = RecursiveTextSplitter(chunk_size=8, chunk_overlap=4)
        chunks = splitter.split_text(text)
        assert len(chunks) >= 2

    def test_overlap_content_repeats(self):
        text = "word1 word2 word3 word4 word5 word6"
        splitter = RecursiveTextSplitter(chunk_size=18, chunk_overlap=6)
        chunks = splitter.split_text(text)
        assert len(chunks) >= 2
        for i in range(len(chunks) - 1):
            last_word = chunks[i].split()[-1]
            assert last_word in chunks[i + 1]


class TestEmptySplitFiltering:
    def test_consecutive_separators_produce_no_empty_chunks(self):
        text = "Hello\n\n\n\nWorld"
        splitter = RecursiveTextSplitter(chunk_size=100, chunk_overlap=0)
        chunks = splitter.split_text(text)
        assert all(c.strip() for c in chunks)

    def test_trailing_separator(self):
        text = "Hello\n\n"
        splitter = RecursiveTextSplitter(chunk_size=100, chunk_overlap=0)
        chunks = splitter.split_text(text)
        assert chunks == ["Hello"]


class TestKeepSeparator:
    def test_keep_separator_end_preserves_within_chunk(self):
        text = "A.\n\nB.\n\nC."
        splitter = RecursiveTextSplitter(chunk_size=10, chunk_overlap=0, keep_separator="end")
        chunks = splitter.split_text(text)
        assert any("\n\n" in c for c in chunks)

    def test_keep_separator_start_preserves_within_chunk(self):
        text = "A.\n\nB.\n\nC."
        splitter = RecursiveTextSplitter(chunk_size=10, chunk_overlap=0, keep_separator="start")
        chunks = splitter.split_text(text)
        assert any("\n\n" in c for c in chunks)

    def test_keep_separator_false_uses_separator_as_joiner(self):
        text = "AA.\n\nBB.\n\nCC."
        splitter = RecursiveTextSplitter(chunk_size=10, chunk_overlap=0, keep_separator=False)
        chunks = splitter.split_text(text)
        assert len(chunks) >= 2
        assert chunks[0] == "AA.\n\nBB."

    def test_no_strip_preserves_boundaries(self):
        text = "First.\n\nSecond.\n\nThird."
        splitter = RecursiveTextSplitter(chunk_size=15, chunk_overlap=0, keep_separator="end", strip_whitespace=False)
        chunks = splitter.split_text(text)
        assert any(c.endswith("\n\n") for c in chunks)


def _word_count(text: str) -> int:
    return len(text.split())


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class TestLengthFunction:
    def test_custom_word_count(self):
        text = "one two three four five six seven eight nine ten"
        splitter = RecursiveTextSplitter(chunk_size=4, chunk_overlap=0, length_function=_word_count)
        chunks = splitter.split_text(text)
        assert len(chunks) >= 2
        assert all(_word_count(c) <= 4 for c in chunks)

    def test_token_approximation(self):
        text = "The quick brown fox jumps over the lazy dog. " * 10
        splitter = RecursiveTextSplitter(chunk_size=20, chunk_overlap=0, length_function=_approx_tokens)
        chunks = splitter.split_text(text)
        assert len(chunks) >= 2
        assert all(_approx_tokens(c) <= 20 for c in chunks)


class TestRecursion:
    def test_large_piece_recurses_to_finer_separator(self):
        text = "Short.\n\n" + "A very long paragraph with many words that exceeds the chunk size limit easily. " * 3
        splitter = RecursiveTextSplitter(chunk_size=50, chunk_overlap=0)
        chunks = splitter.split_text(text)
        assert all(len(c) <= 50 or " " not in c for c in chunks)

    def test_mixed_small_and_large_pieces(self):
        text = "Small.\n\n" + "x" * 200 + "\n\nSmall again."
        splitter = RecursiveTextSplitter(chunk_size=50, chunk_overlap=0)
        chunks = splitter.split_text(text)
        assert "Small." in chunks[0]
        assert "Small again." in chunks[-1]
