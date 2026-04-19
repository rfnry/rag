from rfnry_rag.retrieval.common.models import RetrievedChunk


def format_chunk_header(chunk: RetrievedChunk) -> str:
    """Format a chunk's source attribution header."""
    source_name = chunk.source_metadata.get("name", "")
    page_ref = f"Page {chunk.page_number}" if chunk.page_number else "Unknown page"
    section = f" - {chunk.section}" if chunk.section else ""
    if source_name:
        return f"[{source_name} — {page_ref}{section}]"
    return f"[{page_ref}{section}]"


def chunks_to_context(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks into a context string with source attribution headers."""
    parts = [f"{format_chunk_header(chunk)}\n{chunk.content}" for chunk in chunks]
    return "\n\n---\n\n".join(parts)
