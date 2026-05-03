"""Contract: BAML prompt bodies (retrieval + reasoning) carry no domain hints."""

from pathlib import Path

_RETRIEVAL_BAML_ROOT = Path("src/rfnry_knowledge/baml/baml_src")
_REASONING_BAML_ROOT = Path("src/rfnry_knowledge/reasoning/baml/baml_src")
_BAML_ROOTS: list[Path] = [_RETRIEVAL_BAML_ROOT, _REASONING_BAML_ROOT]

# Files excluded from the domain-agnostic scan.
# Relative-to-root paths (forward slashes).
_EXCLUDED_RELATIVE: set[str] = {
    # Infrastructure — no prompt bodies at all.
    "clients.baml",
    "generators.baml",
    # Drawing ingestion prompt: this file is exempt because the drawing
    # pipeline is inherently domain-tied via DrawingIngestionConfig
    # (default_domain, symbol_library, off_page_connector_patterns). Two
    # categories of bias-listed terms appear here legitimately:
    #   1. enum-bound labels in @description annotations and the prompt's
    #      DOMAIN HINT enumeration ("electrical | p_and_id | mechanical |
    #      mixed") — these are config-bound vocabulary, not prompt examples;
    #   2. domain-flavoured illustrative examples in the SynthesizeDrawingSet
    #      prompt body ("valve V-101 is documented on sheet 5", "wire/line
    #      between components") — retained because the prompt's contrast
    #      between deterministic wire connections and narrative cross-
    #      references is clearer with concrete examples than with abstract
    #      placeholders.
    # If the drawing pipeline is ever generalised away from its
    # electrical/P&ID/mechanical defaults, revisit this exclusion.
    "ingestion/drawing.baml",
}


def _discover_prompt_files() -> list[Path]:
    files: list[Path] = []
    for root in _BAML_ROOTS:
        for p in sorted(root.rglob("*.baml")):
            rel = p.relative_to(root).as_posix()
            if rel in _EXCLUDED_RELATIVE:
                continue
            files.append(p)
    return files


_PROMPT_FILES: list[Path] = _discover_prompt_files()

# Domain-specific words that MUST NOT appear in prompts — they bias the LLM
# toward electrical/mechanical output. (Case-insensitive substring check.)
_DOMAIN_BIAS_TERMS = [
    "valve",
    "motor",
    "wire",
    "terminal",
    "480V",
    "PSI",
    "RPM",
    "SAE",
    "RV-2201",
    "electrical",
    "mechanical",
]


def _extract_prompt_bodies(text: str) -> list[str]:
    """Yield each #" ... "# block from a .baml file."""
    out: list[str] = []
    i = 0
    while True:
        start = text.find('#"', i)
        if start == -1:
            break
        end = text.find('"#', start + 2)
        if end == -1:
            break
        out.append(text[start + 2 : end])
        i = end + 2
    return out


def _display_path(prompt_file: Path) -> str:
    for root in _BAML_ROOTS:
        try:
            return prompt_file.relative_to(root.parent.parent).as_posix()
        except ValueError:
            continue
    return str(prompt_file)


def test_baml_prompts_have_no_domain_bias_terms() -> None:
    assert _PROMPT_FILES, "no BAML prompt files discovered"
    violations: list[str] = []
    for prompt_file in _PROMPT_FILES:
        text = prompt_file.read_text()
        prompt_bodies = _extract_prompt_bodies(text)
        combined = " ".join(prompt_bodies).lower()
        hits = [term for term in _DOMAIN_BIAS_TERMS if term.lower() in combined]
        if hits:
            violations.append(f"{_display_path(prompt_file)}: {hits}")
    assert not violations, (
        "Prompt bodies contain domain-bias terms:\n  "
        + "\n  ".join(violations)
        + "\nThese must be removed to keep the SDK consumer-agnostic."
    )
