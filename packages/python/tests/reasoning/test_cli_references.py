import os
from pathlib import Path

import click
import pytest

from rfnry_rag.reasoning.cli import _MAX_DIR_READ_BYTES, _read_directory_as_text, resolve_references


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


class TestReadDirectoryAsText:
    def test_reasoning_cli_directory_read_caps_aggregate_size(self, tmp_path):
        big = tmp_path / "big.txt"
        big.write_text("x" * (_MAX_DIR_READ_BYTES + 1))
        with pytest.raises(ValueError, match="exceeds"):
            _read_directory_as_text(tmp_path)

    def test_reads_within_limit(self, tmp_path):
        (tmp_path / "a.md").write_text("small content")
        result = _read_directory_as_text(tmp_path)
        assert "small content" in result

    def test_skips_non_text_files(self, tmp_path):
        (tmp_path / "code.py").write_text("print('hello')")
        (tmp_path / "doc.md").write_text("markdown")
        result = _read_directory_as_text(tmp_path)
        assert "markdown" in result
        assert "code.py" not in result

    def test_read_directory_rejects_symlink_outside(self, tmp_path: Path) -> None:
        inside = tmp_path / "inside"
        inside.mkdir()
        (inside / "safe.txt").write_text("ok")

        outside = tmp_path / "outside.txt"
        outside.write_text("sensitive")

        # Symlink inside the dir pointing to a file outside
        link = inside / "evil.txt"
        os.symlink(outside, link)

        with pytest.raises(ValueError, match="escapes directory"):
            _read_directory_as_text(inside)
