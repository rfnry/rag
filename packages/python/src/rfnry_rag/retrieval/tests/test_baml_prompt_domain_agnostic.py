"""Contract: BAML prompt bodies (ingestion + retrieval) carry no domain hints."""
from pathlib import Path

_BAML_SRC_ROOT = Path(__file__).parent.parent / "baml" / "baml_src"

# Every prompt-body file that should be screened for domain-bias terms.
# Add new .baml files here when they land.
_PROMPT_FILES: list[Path] = [
    _BAML_SRC_ROOT / "ingestion" / "functions.baml",
    _BAML_SRC_ROOT / "retrieval" / "functions.baml",
]

# Domain-specific words that MUST NOT appear in prompts — they bias the LLM
# toward electrical/mechanical output. (Case-insensitive substring check.)
_DOMAIN_BIAS_TERMS = [
    "valve", "motor", "wire", "terminal",
    "480V", "PSI", "RPM", "SAE", "RV-2201",
    "electrical", "mechanical",
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


def test_baml_prompts_have_no_domain_bias_terms() -> None:
    violations: list[str] = []
    for prompt_file in _PROMPT_FILES:
        assert prompt_file.exists(), f"prompt file missing: {prompt_file}"
        text = prompt_file.read_text()
        # Scan only the prompt bodies (between #" and "#), not field comments or @descriptions
        prompt_bodies = _extract_prompt_bodies(text)
        combined = " ".join(prompt_bodies).lower()
        hits = [term for term in _DOMAIN_BIAS_TERMS if term.lower() in combined]
        if hits:
            violations.append(f"{prompt_file.relative_to(_BAML_SRC_ROOT)}: {hits}")
    assert not violations, (
        "Prompt bodies contain domain-bias terms:\n  "
        + "\n  ".join(violations)
        + "\nThese must be removed to keep the SDK consumer-agnostic."
    )
