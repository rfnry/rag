import click
import pytest

from rfnry_rag.reasoning.cli import resolve_references


class TestResolveReferences:
    def test_single_file(self, tmp_path):
        f = tmp_path / "policy.md"
        f.write_text("Policy content")
        result = resolve_references((str(f),))
        assert "[policy.md]" in result
        assert "Policy content" in result

    def test_multiple_files(self, tmp_path):
        f1 = tmp_path / "a.md"
        f1.write_text("Content A")
        f2 = tmp_path / "b.md"
        f2.write_text("Content B")
        result = resolve_references((str(f1), str(f2)))
        assert "[a.md]" in result
        assert "[b.md]" in result
        assert "========" in result

    def test_folder(self, tmp_path):
        (tmp_path / "one.md").write_text("One")
        (tmp_path / "two.txt").write_text("Two")
        (tmp_path / "skip.py").write_text("not included")
        result = resolve_references((str(tmp_path),))
        assert "[one.md]" in result
        assert "[two.txt]" in result
        assert "skip.py" not in result

    def test_missing_path_raises(self, tmp_path):
        with pytest.raises(click.UsageError, match="not found"):
            resolve_references((str(tmp_path / "nope.md"),))

    def test_empty_folder_raises(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(click.UsageError, match="No reference documents"):
            resolve_references((str(empty),))
