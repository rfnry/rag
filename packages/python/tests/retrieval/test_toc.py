from unittest.mock import AsyncMock, MagicMock, patch

from rfnry_rag.retrieval.modules.ingestion.tree.toc import (
    PageContent,
    TocInfo,
    TocPath,
    detect_toc,
    find_section_starts,
    parse_toc,
    verify_section_positions,
)

BAML_CLIENT = "rfnry_rag.retrieval.baml.baml_client.async_client.b"


def _make_pages(texts: list[str]) -> list[PageContent]:
    """Helper to create PageContent list from text strings."""
    return [PageContent(index=i, text=t, token_count=len(t.split())) for i, t in enumerate(texts)]


async def test_detect_toc_finds_toc_with_page_numbers():
    """detect_toc returns WITH_PAGE_NUMBERS when a TOC with page numbers is found."""
    pages = _make_pages(
        [
            "Table of Contents\n1. Introduction ... 1\n2. Methods ... 5",
            "Some body text about the project.",
        ]
    )

    mock_result_toc = MagicMock()
    mock_result_toc.has_toc = True
    mock_result_toc.has_page_numbers = True

    mock_result_body = MagicMock()
    mock_result_body.has_toc = False
    mock_result_body.has_page_numbers = False

    with patch(BAML_CLIENT) as mock_b:
        mock_b.DetectTableOfContents = AsyncMock(side_effect=[mock_result_toc, mock_result_body])

        result = await detect_toc(pages, toc_scan_pages=5, registry=MagicMock())

    assert isinstance(result, TocInfo)
    assert result.path == TocPath.WITH_PAGE_NUMBERS
    assert len(result.toc_pages) == 1
    assert result.toc_pages[0].index == 0


async def test_detect_toc_finds_toc_without_page_numbers():
    """detect_toc returns WITHOUT_PAGE_NUMBERS when TOC lacks page numbers."""
    pages = _make_pages(
        [
            "Contents\n- Introduction\n- Background\n- Methods",
            "Body text here.",
        ]
    )

    mock_result_toc = MagicMock()
    mock_result_toc.has_toc = True
    mock_result_toc.has_page_numbers = False

    mock_result_body = MagicMock()
    mock_result_body.has_toc = False
    mock_result_body.has_page_numbers = False

    with patch(BAML_CLIENT) as mock_b:
        mock_b.DetectTableOfContents = AsyncMock(side_effect=[mock_result_toc, mock_result_body])

        result = await detect_toc(pages, toc_scan_pages=5, registry=MagicMock())

    assert result.path == TocPath.WITHOUT_PAGE_NUMBERS
    assert len(result.toc_pages) == 1


async def test_detect_toc_no_toc_found():
    """detect_toc returns NO_TOC when no pages contain a TOC."""
    pages = _make_pages(
        [
            "This is just regular body text.",
            "More regular body text.",
        ]
    )

    mock_result = MagicMock()
    mock_result.has_toc = False
    mock_result.has_page_numbers = False

    with patch(BAML_CLIENT) as mock_b:
        mock_b.DetectTableOfContents = AsyncMock(return_value=mock_result)

        result = await detect_toc(pages, toc_scan_pages=5, registry=MagicMock())

    assert result.path == TocPath.NO_TOC
    assert result.toc_pages == []


async def test_page_content_dataclass():
    """PageContent dataclass stores index, text, and token_count."""
    page = PageContent(index=3, text="hello world", token_count=2)

    assert page.index == 3
    assert page.text == "hello world"
    assert page.token_count == 2


async def test_verify_section_positions():
    """verify_section_positions calls BAML and marks sections as verified or not."""
    pages = _make_pages(
        [
            "Introduction\nThis is the introduction section.",
            "Methods\nWe describe the methods used.",
            "Results\nHere are the results.",
        ]
    )

    sections = [
        {"structure": "1", "title": "Introduction", "page": 0},
        {"structure": "2", "title": "Methods", "page": 1},
        {"structure": "3", "title": "Results", "page": 2},
    ]

    with patch(BAML_CLIENT) as mock_b:
        mock_b.VerifySectionPosition = AsyncMock(side_effect=[True, True, False])

        result = await verify_section_positions(sections, pages, registry=MagicMock())

    assert len(result) == 3
    assert result[0]["verified"] is True
    assert result[0]["title"] == "Introduction"
    assert result[1]["verified"] is True
    assert result[1]["title"] == "Methods"
    assert result[2]["verified"] is False
    assert result[2]["title"] == "Results"

    assert mock_b.VerifySectionPosition.call_count == 3


async def test_detect_toc_respects_scan_limit():
    """detect_toc only scans up to toc_scan_pages, not the whole document."""
    pages = _make_pages(
        [
            "Page 0 text",
            "Page 1 text",
            "Page 2 text",
            "Page 3 text",
            "Page 4 text",
        ]
    )

    mock_result = MagicMock()
    mock_result.has_toc = False
    mock_result.has_page_numbers = False

    with patch(BAML_CLIENT) as mock_b:
        mock_b.DetectTableOfContents = AsyncMock(return_value=mock_result)

        await detect_toc(pages, toc_scan_pages=2, registry=MagicMock())

    assert mock_b.DetectTableOfContents.call_count == 2


async def test_parse_toc_returns_entries():
    """parse_toc concatenates TOC pages and returns parsed entries."""
    toc_pages = _make_pages(
        [
            "1. Introduction ... 1\n2. Methods ... 5",
        ]
    )

    mock_entry_1 = MagicMock()
    mock_entry_1.structure = "1"
    mock_entry_1.title = "Introduction"
    mock_entry_1.page = 1

    mock_entry_2 = MagicMock()
    mock_entry_2.structure = "2"
    mock_entry_2.title = "Methods"
    mock_entry_2.page = 5

    mock_result = MagicMock()
    mock_result.entries = [mock_entry_1, mock_entry_2]

    with patch(BAML_CLIENT) as mock_b:
        mock_b.ParseTableOfContents = AsyncMock(return_value=mock_result)

        entries = await parse_toc(toc_pages, registry=MagicMock())

    assert len(entries) == 2
    assert entries[0] == {"structure": "1", "title": "Introduction", "page": 1}
    assert entries[1] == {"structure": "2", "title": "Methods", "page": 5}


async def test_find_section_starts_fills_missing_pages():
    """find_section_starts calls BAML FindSectionStart for entries without pages."""
    pages = _make_pages(["Intro text", "Methods text", "Results text"])

    entries = [
        {"structure": "1", "title": "Introduction", "page": None},
        {"structure": "2", "title": "Methods", "page": None},
    ]

    with patch(BAML_CLIENT) as mock_b:
        mock_b.FindSectionStart = AsyncMock(side_effect=[0, 1])

        result = await find_section_starts(entries, pages, registry=MagicMock())

    assert len(result) == 2
    assert result[0]["page"] == 0
    assert result[1]["page"] == 1
    assert mock_b.FindSectionStart.call_count == 2


async def test_find_section_starts_skips_entries_with_page():
    """find_section_starts preserves entries that already have a page number."""
    pages = _make_pages(["Intro text"])

    entries = [
        {"structure": "1", "title": "Introduction", "page": 3},
    ]

    with patch(BAML_CLIENT) as mock_b:
        mock_b.FindSectionStart = AsyncMock()

        result = await find_section_starts(entries, pages, registry=MagicMock())

    assert result[0]["page"] == 3
    mock_b.FindSectionStart.assert_not_called()


async def test_verify_section_positions_handles_missing_page():
    """verify_section_positions marks sections with no page as unverified."""
    pages = _make_pages(["Some text"])

    sections = [
        {"structure": "1", "title": "Missing", "page": None},
    ]

    with patch(BAML_CLIENT) as mock_b:
        mock_b.VerifySectionPosition = AsyncMock()

        result = await verify_section_positions(sections, pages, registry=MagicMock())

    assert result[0]["verified"] is False
    mock_b.VerifySectionPosition.assert_not_called()
