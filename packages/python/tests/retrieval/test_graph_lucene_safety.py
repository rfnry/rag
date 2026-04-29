import pytest

from rfnry_rag.retrieval.stores.graph.neo4j import _escape_lucene_query


class TestLuceneEscape:
    def test_plain_text_unchanged(self) -> None:
        assert _escape_lucene_query("plain text") == "plain text"

    @pytest.mark.parametrize("char", list('+-&|!(){}[]^"~*?:\\/'))
    def test_metacharacter_escaped(self, char: str) -> None:
        assert _escape_lucene_query(f"a{char}b") == f"a\\{char}b"

    def test_wildcard_neutralised(self) -> None:
        # Lucene treats bare * as match-all — escape must prevent this.
        assert _escape_lucene_query("*") == r"\*"
        assert _escape_lucene_query("star*power") == r"star\*power"

    def test_field_syntax_neutralised(self) -> None:
        # "name:value" is a Lucene field match — escape the colon.
        assert _escape_lucene_query("name:value") == r"name\:value"

    def test_quote_escaped(self) -> None:
        assert _escape_lucene_query('q"note') == r"q\"note"

    def test_empty_string(self) -> None:
        assert _escape_lucene_query("") == ""
