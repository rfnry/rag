from unittest.mock import patch

from rfnry_knowledge.cli.output import OutputMode, get_output_mode, print_health, print_json
from rfnry_knowledge.models import HealthSummary, RetrievalHealth, Source


class TestGetOutputMode:
    def test_explicit_json(self):
        assert get_output_mode("json") == OutputMode.JSON

    def test_explicit_pretty(self):
        assert get_output_mode("pretty") == OutputMode.PRETTY

    def test_tty_returns_pretty(self):
        with patch("rfnry_knowledge.cli.output.sys.stdout") as mock_stdout:
            mock_stdout.isatty.return_value = True
            assert get_output_mode(None) == OutputMode.PRETTY

    def test_pipe_returns_json(self):
        with patch("rfnry_knowledge.cli.output.sys.stdout") as mock_stdout:
            mock_stdout.isatty.return_value = False
            assert get_output_mode(None) == OutputMode.JSON


class TestPrintJson:
    def test_prints_dataclass(self, capsys):
        source = Source(source_id="abc", chunk_count=10, embedding_model="openai:text-embedding-3-small")
        print_json(source)
        output = capsys.readouterr().out
        assert '"source_id": "abc"' in output
        assert '"chunk_count": 10' in output

    def test_prints_dict(self, capsys):
        print_json({"key": "value"})
        output = capsys.readouterr().out
        assert '"key": "value"' in output


class TestPrintHealth:
    def test_clean_source(self, capsys):
        h = HealthSummary(
            source_id="abc",
            fully_ingested=True,
            ingestion_notes=[],
            stale_embedding=False,
            embedding_model="openai:text-embedding-3-small",
            retrieval=None,
        )
        print_health(h)
        out = capsys.readouterr().out
        assert "abc" in out
        assert "Ingested: ok" in out
        assert "fresh" in out
        assert "no hits recorded" in out

    def test_with_notes_and_retrieval(self, capsys):
        h = HealthSummary(
            source_id="xyz",
            fully_ingested=False,
            ingestion_notes=["vision:warn:page_3:bad"],
            stale_embedding=True,
            embedding_model="openai:text-embedding-3-small",
            retrieval=RetrievalHealth(
                total_hits=10,
                grounded_hits=7,
                ungrounded_hits=3,
                grounding_rate=0.7,
            ),
        )
        print_health(h)
        out = capsys.readouterr().out
        assert "1 note" in out
        assert "vision:warn:page_3:bad" in out
        assert "stale" in out
        assert "10 hits" in out
        assert "70% grounded" in out
