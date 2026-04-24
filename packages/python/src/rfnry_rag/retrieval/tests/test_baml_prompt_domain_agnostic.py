"""Contract: AnalyzePage + ExtractEntitiesFromText prompts carry no domain hints."""
from pathlib import Path

_PROMPT_FILE = (
    Path(__file__).parent.parent
    / "baml" / "baml_src" / "ingestion" / "functions.baml"
)

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


def test_analyze_page_prompt_has_no_domain_bias_terms() -> None:
    text = _PROMPT_FILE.read_text()
    # Scan only the prompt bodies (between #" and "#), not field comments or @descriptions
    prompt_bodies = _extract_prompt_bodies(text)
    combined = " ".join(prompt_bodies).lower()
    violations = [term for term in _DOMAIN_BIAS_TERMS if term.lower() in combined]
    assert not violations, (
        f"Prompt bodies in functions.baml contain domain-bias terms {violations}. "
        "These must be removed to keep the SDK consumer-agnostic."
    )
