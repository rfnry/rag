"""Regression: XML/L5X parsers must not resolve external entities.

Bare lxml `etree.parse` / `etree.iterparse` resolve `file://` entities by default.
A malicious XML/L5X file containing an entity declaration pointing at a local file
could exfiltrate secrets through parsed output. These tests exercise that vector
and assert the parsers are hardened."""

import textwrap
from pathlib import Path

from rfnry_rag.ingestion.analyze.parsers.l5x.parser import parse_l5x
from rfnry_rag.ingestion.analyze.parsers.xml import parse_xml


def _write_xxe_file(tmp_path: Path, root: str, secret: Path, suffix: str = ".xml") -> Path:
    payload = textwrap.dedent(
        f"""<?xml version="1.0"?>
        <!DOCTYPE r [
          <!ENTITY xxe SYSTEM "file://{secret}">
        ]>
        <{root}><val>&xxe;</val></{root}>
        """
    ).strip()
    p = tmp_path / f"{root}{suffix}"
    p.write_text(payload)
    return p


def test_xml_parser_does_not_resolve_external_entities(tmp_path: Path) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP-SECRET-CONTENT")
    xml_file = _write_xxe_file(tmp_path, "root", secret)

    try:
        result = parse_xml(xml_file)
    except Exception as exc:
        assert "TOP-SECRET-CONTENT" not in str(exc)
        return

    serialized = repr(result)
    assert "TOP-SECRET-CONTENT" not in serialized


def test_l5x_parser_does_not_resolve_external_entities(tmp_path: Path) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP-SECRET-CONTENT")
    l5x_file = _write_xxe_file(tmp_path, "RSLogix5000Content", secret, suffix=".l5x")

    try:
        result = parse_l5x(l5x_file)
        assert "TOP-SECRET-CONTENT" not in repr(result)
    except Exception as exc:
        assert "TOP-SECRET-CONTENT" not in str(exc)
