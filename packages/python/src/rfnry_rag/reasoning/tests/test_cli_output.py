from unittest.mock import patch

from rfnry_rag.reasoning.cli.output import OutputMode, get_output_mode, print_json
from rfnry_rag.reasoning.modules.analysis.models import AnalysisResult


class TestGetOutputMode:
    def test_explicit_json(self):
        assert get_output_mode("json") == OutputMode.JSON

    def test_explicit_pretty(self):
        assert get_output_mode("pretty") == OutputMode.PRETTY

    def test_tty_returns_pretty(self):
        with patch("sys.stdout") as mock_stdout:
            mock_stdout.isatty.return_value = True
            assert get_output_mode(None) == OutputMode.PRETTY

    def test_pipe_returns_json(self):
        with patch("sys.stdout") as mock_stdout:
            mock_stdout.isatty.return_value = False
            assert get_output_mode(None) == OutputMode.JSON


class TestPrintJson:
    def test_prints_dataclass(self, capsys):
        result = AnalysisResult(primary_intent="test", confidence=0.9)
        print_json(result)
        output = capsys.readouterr().out
        assert '"primary_intent": "test"' in output
        assert '"confidence": 0.9' in output

    def test_prints_dict(self, capsys):
        print_json({"key": "value"})
        output = capsys.readouterr().out
        assert '"key": "value"' in output
