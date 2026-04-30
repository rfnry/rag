from unittest.mock import patch

from rfnry_rag.cli.output import OutputMode, get_output_mode, print_json
from rfnry_rag.retrieval.common.models import Source


class TestGetOutputMode:
    def test_explicit_json(self):
        assert get_output_mode("json") == OutputMode.JSON

    def test_explicit_pretty(self):
        assert get_output_mode("pretty") == OutputMode.PRETTY

    def test_tty_returns_pretty(self):
        with patch("rfnry_rag.cli.output.sys.stdout") as mock_stdout:
            mock_stdout.isatty.return_value = True
            assert get_output_mode(None) == OutputMode.PRETTY

    def test_pipe_returns_json(self):
        with patch("rfnry_rag.cli.output.sys.stdout") as mock_stdout:
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
