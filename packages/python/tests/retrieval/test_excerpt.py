from rfnry_knowledge.stores.document.excerpt import extract_window


def test_extract_window_centers_on_match():
    content = (
        "Introduction to the system.\n\n"
        "Section 2: Specifications\n"
        "The FBD-20254-MERV13 filter has a pressure drop of 0.25 inches WG.\n"
        "Operating temperature range: -20F to 200F.\n\n"
        "Section 3: Installation\n"
        "Mount the filter with arrows pointing in the direction of airflow."
    )
    excerpt = extract_window(content, "FBD-20254", window_chars=200)
    assert "FBD-20254" in excerpt
    assert "pressure drop" in excerpt


def test_extract_window_at_start():
    content = "FBD-20254 is the primary filter model. It handles high CFM."
    excerpt = extract_window(content, "FBD-20254", window_chars=200)
    assert excerpt.startswith("FBD-20254")
    assert "..." not in excerpt[:10]


def test_extract_window_no_match():
    content = "Some unrelated content about air filters."
    excerpt = extract_window(content, "NONEXISTENT", window_chars=200)
    assert excerpt == content


def test_extract_window_truncates_long_content():
    content = "A" * 500 + " TARGET_TERM " + "B" * 500
    excerpt = extract_window(content, "TARGET_TERM", window_chars=200)
    assert "TARGET_TERM" in excerpt
    assert excerpt.startswith("...")
    assert excerpt.endswith("...")
    assert len(excerpt) < len(content)
