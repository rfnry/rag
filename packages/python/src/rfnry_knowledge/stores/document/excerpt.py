def extract_window(content: str, query: str, window_chars: int = 800) -> str:
    """Extract a text window around the first occurrence of query in content."""
    lower_content = content.lower()
    lower_query = query.lower()
    pos = lower_content.find(lower_query)

    if pos == -1:
        return content[:window_chars] if len(content) > window_chars else content

    half = window_chars // 2
    start = max(0, pos - half)
    end = min(len(content), pos + len(query) + half)

    para_start = content.rfind("\n\n", 0, start)
    if para_start != -1 and (start - para_start) < half:
        start = para_start + 2

    para_end = content.find("\n\n", end)
    if para_end != -1 and (para_end - end) < half:
        end = para_end

    excerpt = content[start:end].strip()
    if start > 0:
        excerpt = "..." + excerpt
    if end < len(content):
        excerpt = excerpt + "..."
    return excerpt
