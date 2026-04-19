def parse_page_range(page_range: str) -> set[int]:
    """Parse a page range string like '5-10,15-20' into a set of page numbers.

    Supports single pages ('5'), ranges ('5-10'), and comma-separated combinations ('1-3,8,10-12').
    All page numbers must be positive integers (1-based).

    Raises ValueError on invalid input.
    """
    page_range = page_range.strip()
    if not page_range:
        raise ValueError("Page range cannot be empty")

    pages: set[int] = set()

    for part in page_range.split(","):
        part = part.strip()
        if not part:
            continue

        if "-" in part and not part.startswith("-"):
            pieces = part.split("-", maxsplit=1)
            try:
                start = int(pieces[0].strip())
                end = int(pieces[1].strip())
            except ValueError:
                raise ValueError(f"Invalid page range segment: '{part}'") from None

            if start < 1 or end < 1:
                raise ValueError(f"Page numbers must be positive, got: '{part}'")
            if start > end:
                raise ValueError(f"Range start ({start}) must not exceed end ({end})")

            pages.update(range(start, end + 1))
        else:
            try:
                page = int(part)
            except ValueError:
                raise ValueError(f"Invalid page number: '{part}'") from None

            if page < 1:
                raise ValueError(f"Page numbers must be positive, got: {page}")
            pages.add(page)

    return pages
