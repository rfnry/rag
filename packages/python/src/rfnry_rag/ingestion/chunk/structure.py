"""Structure-aware preprocessing: detect atomic regions + section headings."""

from __future__ import annotations

import re
from dataclasses import dataclass

_CODE_FENCE_RE = re.compile(r"```[a-zA-Z0-9_+-]*\n.*?\n```", re.DOTALL)
_TABLE_RE = re.compile(
    r"(?:^\|[^\n]+\|\n)"  # header row
    r"(?:^\|[\s\-:|]+\|\n)"  # separator row
    r"(?:^\|[^\n]+\|\n?)+",  # body rows
    re.MULTILINE,
)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)$", re.MULTILINE)


@dataclass
class AtomicRegion:
    start: int
    end: int
    kind: str  # "code_fence" | "table"
    content: str


@dataclass
class HeadingSpan:
    """Heading hierarchy valid from ``start`` char offset to ``end``."""

    start: int
    end: int
    path: tuple[str, ...]  # e.g. ("Safety", "Lockout procedures", "Step 2")


def find_atomic_regions(text: str) -> list[AtomicRegion]:
    regions: list[AtomicRegion] = []
    for m in _CODE_FENCE_RE.finditer(text):
        regions.append(AtomicRegion(m.start(), m.end(), "code_fence", m.group(0)))
    for m in _TABLE_RE.finditer(text):
        regions.append(AtomicRegion(m.start(), m.end(), "table", m.group(0)))
    regions.sort(key=lambda r: r.start)
    return regions


def build_heading_spans(
    text: str,
    exclude_regions: list[AtomicRegion] | None = None,
) -> list[HeadingSpan]:
    """Return non-overlapping spans with the heading path active at each offset.

    When ``exclude_regions`` is provided, heading-like lines that fall inside
    any of those regions (e.g. inside fenced code blocks) are NOT treated as
    real headings. This prevents Python ``# comments`` inside code fences
    from being lifted to section labels.
    """
    headings: list[tuple[int, int, str]] = []  # (offset, level, title)
    exclude = exclude_regions or []
    for m in _HEADING_RE.finditer(text):
        offset = m.start()
        if any(r.start <= offset < r.end for r in exclude):
            continue
        title = m.group(2).strip()
        # Strip optional CommonMark closing sequence: zero or more '#' preceded by whitespace
        title = re.sub(r"\s*#+\s*$", "", title).strip() or title
        headings.append((offset, len(m.group(1)), title))
    if not headings:
        return []
    spans: list[HeadingSpan] = []
    stack: list[tuple[int, str]] = []  # (level, title)
    for i, (offset, level, title) in enumerate(headings):
        # Close headings of equal or deeper level
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        end = headings[i + 1][0] if i + 1 < len(headings) else len(text)
        spans.append(
            HeadingSpan(
                start=offset,
                end=end,
                path=tuple(t for _, t in stack),
            )
        )
    return spans


def section_path_at(offset: int, spans: list[HeadingSpan]) -> str | None:
    """Return 'A > B > C' heading path for the given offset, or None."""
    for s in spans:
        if s.start <= offset < s.end:
            return " > ".join(s.path)
    return None
