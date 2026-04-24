"""TOC detection, parsing, and section position verification."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Any

from rfnry_rag.retrieval.common.logging import get_logger

logger = get_logger(__name__)


class TocPath(Enum):
    WITH_PAGE_NUMBERS = "with_page_numbers"
    WITHOUT_PAGE_NUMBERS = "without_page_numbers"
    NO_TOC = "no_toc"


@dataclass
class PageContent:
    index: int
    text: str
    token_count: int


@dataclass
class TocInfo:
    path: TocPath
    toc_pages: list[PageContent]


async def detect_toc(
    pages: list[PageContent],
    toc_scan_pages: int,
    registry: Any,
) -> TocInfo:
    """Scan the first N pages for a table of contents.

    Calls BAML DetectTableOfContents on each page within the scan range and
    returns a TocInfo describing whether a TOC was found and which path to
    follow (with page numbers, without, or no TOC).
    """
    from rfnry_rag.retrieval.baml.baml_client.async_client import b

    scan_limit = min(toc_scan_pages, len(pages))
    candidates = pages[:scan_limit]

    async def _detect_page(page: PageContent) -> tuple[PageContent, Any]:
        result = await b.DetectTableOfContents(
            page_text=page.text,
            baml_options={"client_registry": registry},
        )
        return page, result

    results = await asyncio.gather(*[_detect_page(p) for p in candidates])

    toc_pages: list[PageContent] = []
    has_page_numbers = False

    for page, result in results:
        if result.has_toc:
            toc_pages.append(page)
            if result.has_page_numbers:
                has_page_numbers = True

    if not toc_pages:
        logger.debug("no TOC detected in first %d pages", scan_limit)
        return TocInfo(path=TocPath.NO_TOC, toc_pages=[])

    path = TocPath.WITH_PAGE_NUMBERS if has_page_numbers else TocPath.WITHOUT_PAGE_NUMBERS
    logger.debug("TOC detected: path=%s, toc_pages=%d", path.value, len(toc_pages))
    return TocInfo(path=path, toc_pages=toc_pages)


async def parse_toc(
    toc_pages: list[PageContent],
    registry: Any,
) -> list[dict[str, Any]]:
    """Parse TOC pages into structured section entries.

    Concatenates the text of all TOC pages and calls BAML ParseTableOfContents.
    Returns a list of dicts with keys: structure, title, page (optional).
    """
    from rfnry_rag.retrieval.baml.baml_client.async_client import b

    combined_text = "\n".join(page.text for page in toc_pages)

    result = await b.ParseTableOfContents(
        toc_text=combined_text,
        baml_options={"client_registry": registry},
    )

    entries = []
    for entry in result.entries:
        entries.append(
            {
                "structure": entry.structure,
                "title": entry.title,
                "page": entry.page,
            }
        )

    logger.debug("parsed %d TOC entries", len(entries))
    return entries


async def find_section_starts(
    entries: list[dict[str, Any]],
    pages: list[PageContent],
    registry: Any,
) -> list[dict[str, Any]]:
    """For TOC entries without page numbers, find where each section starts.

    Calls BAML FindSectionStart for each entry to locate the page where
    that section heading appears. Returns updated entries with page numbers.
    """
    from rfnry_rag.retrieval.baml.baml_client.async_client import b

    # Build pages text in groups to avoid sending entire document per call.
    # Each group covers ~50 pages, which keeps context manageable.
    group_size = 50
    page_groups: list[str] = []
    for i in range(0, len(pages), group_size):
        group = pages[i : i + group_size]
        page_groups.append("\n".join(f"--- Page {page.index} ---\n{page.text}" for page in group))

    updated: list[dict[str, Any]] = []
    for entry in entries:
        if entry.get("page") is not None:
            updated.append(entry)
            continue

        # SERIAL: early-exit search — groups are tried in order and the loop
        # breaks as soon as the section start is found. Parallel execution would
        # waste LLM calls on groups after the match and cannot short-circuit.
        page_num = None
        for group_text in page_groups:
            page_num = await b.FindSectionStart(
                section_title=entry["title"],
                pages_text=group_text,
                baml_options={"client_registry": registry},
            )
            if page_num is not None and page_num > 0:
                break

        updated.append(
            {
                **entry,
                "page": page_num,
            }
        )
        logger.debug("found start for '%s' at page %s", entry["title"], page_num)

    return updated


async def verify_section_positions(
    sections: list[dict[str, Any]],
    pages: list[PageContent],
    registry: Any,
) -> list[dict[str, Any]]:
    """Verify that each section heading actually appears on its claimed page.

    Runs BAML VerifySectionPosition concurrently for all sections. Sections
    that fail verification are marked with verified=False.
    """
    from rfnry_rag.retrieval.baml.baml_client.async_client import b

    page_by_index = {page.index: page for page in pages}

    async def _verify(section: dict[str, Any]) -> dict[str, Any]:
        page_index = section.get("page")
        if page_index is None or page_index not in page_by_index:
            return {**section, "verified": False}

        page = page_by_index[page_index]
        verified = await b.VerifySectionPosition(
            title=section["title"],
            page_text=page.text,
            baml_options={"client_registry": registry},
        )

        return {**section, "verified": verified}

    tasks = [asyncio.create_task(_verify(section)) for section in sections]
    results = list(await asyncio.gather(*tasks))

    verified_count = sum(1 for r in results if r.get("verified"))
    logger.debug("verified %d/%d section positions", verified_count, len(results))

    return results
